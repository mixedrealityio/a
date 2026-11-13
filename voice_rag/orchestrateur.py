"""Orchestrateur vocal RAG.

    uvicorn voice_rag.orchestrateur:app --host 0.0.0.0 --port 8000

WebSocket /ws
  client -> serveur : trames binaires = PCM 16 bits mono (micro) ; trames texte = événements JSON
      session.start {user_id, sample_rate, vad, barge_in}    text {question}      turn.end
      image {data, mime}      video.frame {data}      video.stop      menu.select {index}      interrupt
  serveur -> client : trames binaires = PCM 16 bits mono de la réponse ; trames texte = événements JSON
      session.ready, state, transcript, rag, answer.delta, answer.card, audio.end, interrupt, metrics, error
POST /texte : même pipeline sans audio, pour les tests.
"""
import asyncio
import io
import json
import logging
import time
import wave
from collections import deque
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import prompts as P
from .config import CFG
from .controle import (ContexteControle, Decoupeur, decouper, nettoyer_pour_voix, reperes_utilises,
                       verifier_phrase)
from .memoire import Memoire
from .qwen import QwenClient, contenu_audio, contenu_media
from .rag_client import RagClient, ResultatRag

log = logging.getLogger("orchestrateur")
TRANSPORT_HTTP = None          # remplacé dans les tests pour joindre les faux backends sans réseau


class Services:
    http: httpx.AsyncClient
    qwen: QwenClient
    rag: RagClient
    mem: Memoire


S = Services()


@asynccontextmanager
async def cycle_de_vie(app):
    limites = httpx.Limits(max_keepalive_connections=20, keepalive_expiry=60)
    S.http = httpx.AsyncClient(limits=limites, transport=TRANSPORT_HTTP)   # connexions gardées ouvertes
    S.qwen, S.rag, S.mem = QwenClient(S.http), RagClient(S.http), Memoire()
    yield
    await S.http.aclose()


app = FastAPI(title="Orchestrateur vocal RAG", lifespan=cycle_de_vie)


def pcm_vers_wav(pcm: bytes, hz: int) -> bytes:
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(hz)
        w.writeframes(pcm)
    return tampon.getvalue()


# --------------------------------------------------------------------------- session
class Session:
    """État d'une connexion. `envoyer_json` / `envoyer_audio` sont les deux seuls points de sortie."""

    def __init__(self, ws: WebSocket | None = None):
        self.ws, self.verrou = ws, asyncio.Lock()
        self.user, self.hz, self.vad, self.barge_in = "anonyme", 16000, True, False
        self.images: list[tuple[str, str]] = []
        self.frames: deque = deque(maxlen=CFG.video_max_frames)
        self.tour: asyncio.Task | None = None
        self.evenements: list[dict] = []          # utilisé par POST /texte
        self.voix_active = ws is not None
        self._reste, self._pre, self._pre_ms = b"", deque(), 0.0
        self._voix, self._actif_ms, self._silence_ms, self._parole = bytearray(), 0.0, 0.0, False

    async def envoyer_json(self, type_: str, **champs):
        msg = {"type": type_, **champs}
        if self.ws is None:
            self.evenements.append(msg)
            return
        async with self.verrou:
            await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    async def envoyer_audio(self, pcm: bytes):
        if self.ws is not None:
            async with self.verrou:
                await self.ws.send_bytes(pcm)

    def tour_en_cours(self) -> bool:
        return self.tour is not None and not self.tour.done()

    # ---- détection de parole par énergie, par trames de 20 ms
    def vad_reinitialiser(self):
        self._reste, self._pre_ms, self._actif_ms, self._silence_ms, self._parole = b"", 0.0, 0.0, 0.0, False
        self._pre.clear()
        self._voix = bytearray()

    def vad_alimenter(self, pcm: bytes) -> list[tuple[str, bytes]]:
        """Retourne les événements ("debut", b"") et ("fin", pcm_de_la_phrase)."""
        sorties, taille = [], int(self.hz * 0.02) * 2
        donnees = self._reste + pcm
        for i in range(0, len(donnees) - taille + 1, taille):
            trame = donnees[i:i + taille]
            a = np.frombuffer(trame, dtype=np.int16).astype(np.float32)
            actif = float(np.sqrt(np.mean(a * a))) >= CFG.vad_seuil
            if not self._parole:
                self._pre.append(trame)
                self._pre_ms += 20
                while self._pre_ms > CFG.vad_preroll_ms:
                    self._pre.popleft()
                    self._pre_ms -= 20
                self._actif_ms = self._actif_ms + 20 if actif else 0.0
                if self._actif_ms >= CFG.vad_debut_ms:
                    self._parole, self._silence_ms = True, 0.0
                    self._voix = bytearray(b"".join(self._pre))
                    self._pre.clear()
                    self._pre_ms = 0.0
                    sorties.append(("debut", b""))
            else:
                self._voix += trame
                self._silence_ms = 0.0 if actif else self._silence_ms + 20
                if self._silence_ms >= CFG.vad_fin_ms or len(self._voix) >= CFG.vad_max_s * self.hz * 2:
                    sorties.append(("fin", bytes(self._voix)))
                    self._parole, self._actif_ms, self._voix = False, 0.0, bytearray()
        self._reste = donnees[len(donnees) - (len(donnees) % taille):] if len(donnees) % taille else b""
        return sorties


# --------------------------------------------------------------------------- sortie vocale
class SortieVocale:
    """Synthétise les phrases validées et envoie l'audio dans l'ordre.

    La synthèse de la phrase suivante démarre pendant que la précédente est encore envoyée.
    """

    def __init__(self, s: Session):
        self.s, self.file = s, asyncio.Queue()
        self.sem = asyncio.Semaphore(CFG.voix_paralleles)
        self.taches: list[asyncio.Task] = []
        self.premier_audio: float | None = None
        self.emetteur = asyncio.create_task(self._emettre())

    def dire(self, phrase: str):
        if not self.s.voix_active or CFG.voix_mode == "aucune" or not phrase:
            return
        blocs: asyncio.Queue = asyncio.Queue()
        self.taches.append(asyncio.create_task(self._synthetiser(phrase, blocs)))
        self.file.put_nowait(blocs)

    async def _synthetiser(self, phrase: str, blocs: asyncio.Queue):
        try:
            async with self.sem:
                async for pcm in S.qwen.dire(phrase):
                    blocs.put_nowait(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:                       # la réponse reste lisible sur la carte
            log.warning("synthèse vocale en échec : %s", e)
        finally:
            blocs.put_nowait(None)

    async def _emettre(self):
        while (blocs := await self.file.get()) is not None:
            while (pcm := await blocs.get()) is not None:
                if self.premier_audio is None:
                    self.premier_audio = time.perf_counter()
                await self.s.envoyer_audio(pcm)

    async def terminer(self):
        self.file.put_nowait(None)
        await self.emetteur

    def annuler(self):
        for t in [*self.taches, self.emetteur]:
            t.cancel()


# --------------------------------------------------------------------------- pipeline d'un tour
def construire_messages(user: str, texte: str | None, wav: bytes | None, images, frames) -> list[dict]:
    messages = [{"role": "system", "content": S.qwen.prompt_systeme(P.PROMPT_SYSTEME)}]
    for t in S.mem.historique(user):
        messages += [{"role": "user", "content": t["question"]}, {"role": "assistant", "content": t["reponse"]}]
    media = contenu_media(images, frames)
    consigne = texte or ""
    menu = S.mem.dernier_menu(user)
    if menu:
        consigne += "\n[Dernier menu proposé : " + " ; ".join(f"{i}. {m}" for i, m in enumerate(menu, 1)) + "]"
    if media and wav and CFG.media_separe:     # une seule modalité non-texte par message utilisateur
        nature = "une vidéo de ma caméra" if frames else "une image"
        messages += [{"role": "user", "content": media + [{"type": "text", "text": f"Voici {nature}. Ma question suit."}]},
                     {"role": "assistant", "content": "Bien reçu."}]
        media = []
    parts = media + ([contenu_audio(wav)] if wav else [])
    if wav and not consigne:
        consigne = "Réponds à la question posée dans l'audio."
    parts.append({"type": "text", "text": consigne.strip()})
    messages.append({"role": "user", "content": parts if len(parts) > 1 else parts[0]["text"]})
    return messages


async def intercepter_recherche(s: Session, args: dict) -> ResultatRag:
    """Exécute l'appel d'outil du modèle : mémoire locale d'abord, backend RAG ensuite."""
    t = time.perf_counter()
    transcription = str(args.get("transcription") or "")
    requete = str(args.get("requete") or args.get("query") or transcription).strip()[:500]
    res, requete = S.mem.chercher(s.user, requete, transcription)
    if res is None:
        try:
            res = await S.rag.rechercher(requete, s.user)
            S.mem.ajouter_rag(s.user, res)
        except Exception as e:
            log.warning("backend RAG en échec : %s", e)
            res = ResultatRag(requete=requete, route="erreur")
    await s.envoyer_json("rag", route=res.route, requete=requete, nb_extraits=len(res.extraits),
                         sources=[e.source for e in res.extraits], menu=res.menu,
                         ms=round((time.perf_counter() - t) * 1000))
    return res


async def manche(messages, *, outils=False, forcer=False, sur_phrase=None):
    """Un appel au modèle. Retourne (texte complet, appels d'outils). `sur_phrase` reçoit les phrases au fil de l'eau."""
    texte, appels, decoupeur = "", [], Decoupeur()
    async for type_, val in S.qwen.flux_avec_repli(messages, outils=outils, forcer=forcer):
        if type_ == "appel":
            appels.append(val)
        elif type_ == "texte":
            texte += val
            if sur_phrase:
                for p in decoupeur.ajouter(val):
                    await sur_phrase(p)
    if sur_phrase:
        for p in decoupeur.vider():
            await sur_phrase(p)
    return texte.strip(), appels


async def traiter_tour(s: Session, *, wav: bytes | None = None, texte: str | None = None):
    t0, mesures = time.perf_counter(), {}
    images, s.images = s.images, []
    frames = [f for ts, f in s.frames if time.time() - ts <= CFG.video_fraicheur_s]
    sortie = SortieVocale(s)
    phrases_ok, problemes, brut = [], [], []
    res = ResultatRag(requete="")
    transcription = texte or ""
    remplacee = False
    try:
        await s.envoyer_json("state", value="reflexion")
        messages = construire_messages(s.user, texte, wav, images, frames)

        async def publier(phrase: str):
            propre = nettoyer_pour_voix(phrase)
            if propre:
                phrases_ok.append(propre)
                if "premiere_phrase_ms" not in mesures:
                    mesures["premiere_phrase_ms"] = round((time.perf_counter() - t0) * 1000)
                    await s.envoyer_json("state", value="parole")
                await s.envoyer_json("answer.delta", text=propre)
                sortie.dire(propre)

        async def controler(phrase: str):
            nonlocal remplacee
            brut.append(phrase)
            ctx = ContexteControle(transcription, res.texte_sources(), len(res.extraits))
            soucis = verifier_phrase(phrase, ctx)
            if not soucis:
                await publier(phrase)
                return
            problemes.extend(soucis)
            log.info("phrase refusée %r : %s", phrase, soucis)
            if not remplacee:                 # une seule phrase de remplacement par tour
                remplacee = True
                await publier(P.REPONSE_NON_CONFIRMEE)

        # 1) le modèle écoute et décide : appel d'outil (cas normal) ou courte phrase sans recherche
        texte1, appels = await manche(messages, outils=True)
        if not appels and (len(texte1) > CFG.sans_rag_max_car or any(c.isdigit() for c in texte1) or not texte1):
            texte1, appels = await manche(messages, outils=True, forcer=True)     # réponse de mémoire : recherche forcée
        mesures["decision_ms"] = round((time.perf_counter() - t0) * 1000)

        # 2) interception des appels d'outils
        final, recherches = (texte1 if not appels else ""), 0
        while appels and recherches < CFG.max_recherches:
            recherches += 1
            messages.append(S.qwen.msg_appels(appels))
            for a in appels:
                if a.nom != P.NOM_OUTIL:
                    messages.append(S.qwen.msg_resultat(a, "ERREUR : outil inconnu."))
                    continue
                r = await intercepter_recherche(s, a.args)
                transcription = transcription or str(a.args.get("transcription") or r.requete)
                if r.extraits or r.menu or not res.extraits:
                    res = r
                messages.append(S.qwen.msg_resultat(a, r.pour_modele()))
            if wav and transcription:
                await s.envoyer_json("transcript", text=transcription)
            mesures["rag_ms"] = round((time.perf_counter() - t0) * 1000)
            if res.route == "erreur":
                final, appels = P.REPONSE_ERREUR, []
                break
            if not res.extraits and not res.menu:          # rien trouvé : pas besoin du modèle
                final, appels = P.REPONSE_INCONNUE, []
                break
            # 3) rédaction à partir des extraits, contrôlée phrase par phrase avant d'être dite
            derniere = recherches >= CFG.max_recherches
            if CFG.controle_mode == "phrase":
                final, appels = await manche(messages, outils=not derniere, sur_phrase=controler)
                if phrases_ok:
                    final, appels = "", []                 # déjà publié
            else:
                final, appels = await manche(messages, outils=not derniere)
        if appels:
            final = final or P.REPONSE_INCONNUE

        # 4) texte restant à contrôler en entier (réponse sans recherche, mode « complet », phrases fixes)
        if final:
            ctx = ContexteControle(transcription, res.texte_sources(), len(res.extraits))
            soucis = [p for ph in decouper(final) for p in verifier_phrase(ph, ctx)]
            if soucis and CFG.controle_mode == "complet" and res.extraits:
                problemes.extend(soucis)
                messages += [{"role": "assistant", "content": final},
                             {"role": "user", "content": P.PROMPT_CORRECTION.format(problemes=" ".join(soucis))}]
                final, _ = await manche(messages)
                soucis = [p for ph in decouper(final) for p in verifier_phrase(ph, ctx)]
            if soucis:
                problemes.extend(soucis)
                final = P.REPONSE_PRUDENTE if res.extraits else P.REPONSE_INCONNUE
            brut.append(final)
            for ph in decouper(final):
                await publier(ph)
        if not phrases_ok:
            await publier(P.REPONSE_INCONNUE)

        await sortie.terminer()
        reponse = " ".join(phrases_ok)
        cites = [res.extraits[i - 1] for i in reperes_utilises(" ".join(brut)) if i <= len(res.extraits)]
        await s.envoyer_json("answer.card", text=reponse, menu=res.menu,
                             sources=[{"id": e.id, "source": e.source} for e in (cites or res.extraits)],
                             controle={"ok": not problemes, "problemes": problemes}, route=res.route)
        await s.envoyer_json("audio.end")
        if sortie.premier_audio:
            mesures["premier_audio_ms"] = round((sortie.premier_audio - t0) * 1000)
        mesures["total_ms"] = round((time.perf_counter() - t0) * 1000)
        await s.envoyer_json("metrics", **mesures)
        S.mem.ajouter_tour(s.user, transcription or "(question audio)", reponse)
        await S.mem.sauver(s.user)
    except asyncio.CancelledError:
        sortie.annuler()
        raise
    except Exception as e:
        log.exception("tour en échec")
        sortie.annuler()
        await s.envoyer_json("error", message=str(e)[:300])
    finally:
        if s.ws is not None:
            try:
                await s.envoyer_json("state", value="ecoute")
            except Exception:
                pass


# --------------------------------------------------------------------------- routes
@app.get("/health")
async def sante():
    return {"ok": True, "modele": CFG.qwen_model, "rag_mode": CFG.rag_mode, "controle": CFG.controle_mode}


@app.post("/texte")
async def question_texte(corps: dict):
    s = Session()
    s.user = str(corps.get("user_id") or "anonyme")
    if corps.get("image"):
        s.images.append((corps.get("mime", "image/jpeg"), corps["image"]))
    await traiter_tour(s, texte=str(corps.get("question", "")))
    return {"evenements": s.evenements}


async def _interrompre(s: Session):
    if s.tour_en_cours():
        s.tour.cancel()
        try:
            await s.tour
        except (asyncio.CancelledError, Exception):
            pass
    await s.envoyer_json("interrupt")


def _lancer(s: Session, **kw):
    s.tour = asyncio.create_task(traiter_tour(s, **kw))


@app.websocket("/ws")
async def flux_voix(ws: WebSocket):
    await ws.accept()
    s = Session(ws)
    appui = bytearray()                                 # audio reçu en mode « appuyer pour parler »
    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                if not s.vad:
                    if not s.tour_en_cours():
                        appui += m["bytes"]
                    continue
                if s.tour_en_cours() and not s.barge_in:
                    continue                             # évite que l'assistant s'entende lui-même
                for evt, pcm in s.vad_alimenter(m["bytes"]):
                    if evt == "debut":
                        if s.barge_in:
                            await _interrompre(s)        # l'utilisateur coupe la parole
                        await s.envoyer_json("state", value="ecoute_active")
                    elif not s.tour_en_cours():
                        _lancer(s, wav=pcm_vers_wav(pcm, s.hz))
                continue
            try:
                e = json.loads(m.get("text") or "{}")
            except ValueError:
                continue
            t = e.get("type")
            if t == "session.start":
                s.user = str(e.get("user_id") or "anonyme")
                s.hz = int(e.get("sample_rate") or 16000)
                s.vad, s.barge_in = bool(e.get("vad", True)), bool(e.get("barge_in", False))
                s.vad_reinitialiser()
                await s.envoyer_json("session.ready", sample_rate_out=CFG.audio_out_hz, user_id=s.user)
            elif t == "image" and e.get("data"):
                s.images = [(e.get("mime", "image/jpeg"), e["data"])]
            elif t == "video.frame" and e.get("data"):
                s.frames.append((time.time(), e["data"]))
            elif t == "video.stop":
                s.frames.clear()
            elif t == "interrupt":
                await _interrompre(s)
            elif t == "turn.end" and appui and not s.tour_en_cours():
                _lancer(s, wav=pcm_vers_wav(bytes(appui), s.hz))
                appui.clear()
            elif t in ("text", "menu.select") and not s.tour_en_cours():
                question = str(e.get("question") or "")
                menu = S.mem.dernier_menu(s.user)
                if t == "menu.select" and 1 <= int(e.get("index", 0)) <= len(menu):
                    question = menu[int(e["index"]) - 1]
                if question:
                    _lancer(s, texte=question)
    except WebSocketDisconnect:
        pass
    finally:
        if s.tour_en_cours():
            s.tour.cancel()
