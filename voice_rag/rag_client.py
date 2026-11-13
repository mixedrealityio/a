"""Client du backend RAG : /chunks quand elle existe, sinon /ask (réponse jetée, data + sources gardées).

Contient aussi le nettoyage des résultats avant envoi au modèle.
"""
import hashlib
import html
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field

import httpx

from .config import CFG

CLES_LISTE = ("chunks", "results", "documents", "data", "contexts", "context", "passages", "hits", "matches")
CLES_TEXTE = ("text", "texte", "content", "contenu", "chunk", "page_content", "passage", "body")
CLES_REPONSE = ("answer", "response", "reponse", "réponse", "output", "generated_text")
CLES_MENU = ("menu", "options", "suggestions", "choices", "follow_ups", "followups", "propositions")
CLES_SCORE = ("score", "similarity", "relevance")
CLES_SOURCE = ("source", "title", "titre", "document", "file", "filename", "name", "nom", "url")


@dataclass
class Extrait:
    id: str
    texte: str
    source: str = ""
    score: float | None = None


@dataclass
class ResultatRag:
    requete: str
    extraits: list[Extrait] = field(default_factory=list)
    menu: list[str] = field(default_factory=list)
    route: str = "ask"            # chunks | ask | memoire | erreur
    ts: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(requete=d["requete"], extraits=[Extrait(**e) for e in d.get("extraits", [])],
                   menu=list(d.get("menu", [])), route=d.get("route", "ask"), ts=d.get("ts", 0.0))

    def pour_modele(self) -> str:
        """Texte compact envoyé au modèle comme résultat de l'outil."""
        if not self.extraits and not self.menu:
            return "AUCUN_RESULTAT"
        lignes = ["<extraits_rag>"]
        for i, e in enumerate(self.extraits, 1):
            src = f" (source : {e.source})" if e.source else ""
            lignes.append(f"[S{i}]{src} {e.texte}")
        lignes.append("</extraits_rag>")
        if self.menu:
            lignes.append("<menu>" + " ; ".join(f"{i}. {m}" for i, m in enumerate(self.menu, 1)) + "</menu>")
        return "\n".join(lignes)

    def texte_sources(self) -> str:
        return " \n".join([e.texte for e in self.extraits] + [e.source for e in self.extraits] + self.menu)


# --------------------------------------------------------------------------- nettoyage
_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮﻿]")
_INJECTION = re.compile(
    r"ignor\w*\s+(?:\w+\s+){0,3}(?:instructions|consignes|r[eè]gles)"
    r"|oubli\w*\s+(?:\w+\s+){0,3}(?:instructions|consignes|r[eè]gles)"
    r"|disregard\s+(?:\w+\s+){0,3}instructions"
    r"|(?:system\s*prompt|prompt\s+syst[eè]me)"
    r"|tu\s+es\s+(?:maintenant|d[ée]sormais)|you\s+are\s+now"
    r"|<\|[^|>]{0,40}\|>|</?tool_(?:call|response)>|</?(?:extraits_rag|menu|lire)>",
    re.I)


def nettoyer_texte(t: str, max_car: int | None = None) -> str:
    """Retire HTML, markdown, caractères invisibles et tentatives d'injection ; tronque à une fin de phrase."""
    max_car = max_car or CFG.rag_max_car_extrait
    t = html.unescape(str(t))
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"</?[a-zA-Z][^>]{0,200}>", " ", t)
    t = _CONTROLE.sub("", t)
    t = re.sub(r"<\|[^|>]{0,40}\|>", " ", t)                 # jetons spéciaux de modèles
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)              # images markdown
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)           # liens markdown -> libellé
    t = re.sub(r"^\s{0,3}>\s*", "", t, flags=re.M)          # citations
    t = re.sub(r"(^|\s)#{1,6}\s+", r"\1", t)                # titres
    t = re.sub(r"^[\s|:\-]{3,}$", " ", t, flags=re.M)        # séparateurs de tableau
    t = re.sub(r"\*\*|__|`+", "", t).replace("|", " ; ")
    morceaux = re.split(r"(?<=[.!?])\s+|\n+", t)
    t = " ".join(m.strip() for m in morceaux if m.strip() and not _INJECTION.search(m))
    t = re.sub(r"\s+", " ", t).strip(" ;")
    if len(t) > max_car:
        coupe = t[:max_car]
        fin = max(coupe.rfind(". "), coupe.rfind("! "), coupe.rfind("? "))
        t = coupe[:fin + 1] if fin > max_car * 0.5 else coupe.rsplit(" ", 1)[0] + "…"
    return t


def _cle_dedoublonnage(t: str) -> str:
    n = unicodedata.normalize("NFKD", t.lower())
    return hashlib.md5(re.sub(r"[^a-z0-9]+", "", n).encode()).hexdigest()


def _libelle_source(o) -> str:
    if o is None:
        return ""
    if isinstance(o, str):
        return nettoyer_texte(o, 120)
    if isinstance(o, dict):
        for k in ("source", "metadata", "meta"):
            if isinstance(o.get(k), dict):
                lib = _libelle_source(o[k])
                if lib:
                    return lib
        nom = next((str(o[k]) for k in CLES_SOURCE if isinstance(o.get(k), (str, int)) and str(o[k]).strip()), "")
        page = o.get("page") or o.get("page_number")
        if nom and page not in (None, ""):
            nom += f", page {page}"
        return nettoyer_texte(nom, 120)
    return ""


def _premier(d: dict, cles):
    return next((d[k] for k in cles if k in d and d[k] not in (None, "", [])), None)


def _liste_items(payload: dict) -> list:
    v = _premier(payload, CLES_LISTE)
    if isinstance(v, dict):                 # ex. {"data": {"chunks": [...]}}
        v = _premier(v, CLES_LISTE) or [v]
    if isinstance(v, str):
        v = [v]
    return v if isinstance(v, list) else []


def extraire_menu(payload: dict) -> list[str]:
    """Menu structuré s'il existe, sinon liste numérotée ou à puces trouvée dans la réponse de /ask."""
    brut = _premier(payload, CLES_MENU)
    if brut is None and isinstance(payload.get("data"), dict):
        brut = _premier(payload["data"], CLES_MENU)
    menu = []
    if isinstance(brut, list):
        for o in brut:
            if isinstance(o, dict):
                o = _premier(o, ("label", "libelle", "title", "titre", "text", "question", "name"))
            if isinstance(o, str) and o.strip():
                menu.append(nettoyer_texte(o, 120))
    if not menu:
        reponse = _premier(payload, CLES_REPONSE)
        if isinstance(reponse, str):
            lignes = re.findall(r"^\s*(?:\d{1,2}[.)]|[-•*])\s+(.{3,120}?)\s*$", reponse, flags=re.M)
            if len(lignes) >= 2:
                menu = [nettoyer_texte(l, 120) for l in lignes]
    return [m for m in menu if m][:6]


def normaliser(payload, requete: str, route: str) -> ResultatRag:
    """Transforme la réponse brute du backend en ResultatRag propre.

    La réponse rédigée par le backend (answer/response/…) n'est jamais copiée : seuls les
    textes de `data`/`chunks` et les `sources` sont gardés. Les embeddings sont ignorés.
    """
    if isinstance(payload, list):
        payload = {"chunks": payload}
    if not isinstance(payload, dict):
        return ResultatRag(requete=requete, route=route)
    sources = payload.get("sources") or payload.get("references") or []
    if not isinstance(sources, list):
        sources = [sources]
    bruts = []
    for i, it in enumerate(_liste_items(payload)):
        if isinstance(it, str):
            texte, score, src, ident = it, None, "", ""
        elif isinstance(it, dict):
            texte = _premier(it, CLES_TEXTE)
            score = _premier(it, CLES_SCORE)
            src = _libelle_source(it)
            ident = str(it.get("id") or it.get("chunk_id") or "")
        else:
            continue
        if not isinstance(texte, str):
            continue
        if not src and i < len(sources):
            src = _libelle_source(sources[i])
        bruts.append((texte, score, src, ident))
    if not bruts:                               # /ask dont les textes sont dans `sources`
        for s in sources:
            if isinstance(s, dict) and isinstance(_premier(s, CLES_TEXTE), str):
                bruts.append((_premier(s, CLES_TEXTE), _premier(s, CLES_SCORE), _libelle_source(s), str(s.get("id") or "")))

    extraits, vus = [], set()
    for texte, score, src, ident in bruts:
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        if score is not None and score < CFG.rag_seuil:
            continue
        propre = nettoyer_texte(texte)
        cle = _cle_dedoublonnage(propre)
        if len(propre) < 15 or cle in vus:
            continue
        vus.add(cle)
        extraits.append(Extrait(id=ident or cle[:8], texte=propre, source=src, score=score))
    extraits.sort(key=lambda e: -(e.score if e.score is not None else 0.0))
    gardes, total = [], 0
    for e in extraits[:CFG.rag_top_k]:
        if total + len(e.texte) > CFG.rag_max_car_total and gardes:
            break
        gardes.append(e)
        total += len(e.texte)
    return ResultatRag(requete=requete, extraits=gardes, menu=extraire_menu(payload), route=route)


# --------------------------------------------------------------------------- client HTTP
class RouteIndisponible(Exception):
    pass


class RagClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self._chunks_ko_depuis: float | None = None

    def _chunks_a_tester(self) -> bool:
        if CFG.rag_mode == "ask":
            return False
        if CFG.rag_mode == "chunks" or self._chunks_ko_depuis is None:
            return True
        return time.monotonic() - self._chunks_ko_depuis > CFG.rag_recheck_s

    async def _post(self, route: str, requete: str, user_id: str):
        entetes = {"Authorization": f"Bearer {CFG.rag_api_key}"} if CFG.rag_api_key else {}
        corps = {CFG.rag_champ_question: requete, "top_k": CFG.rag_top_k, "user_id": user_id}
        r = await self.http.post(CFG.rag_base_url + route, json=corps, headers=entetes, timeout=CFG.rag_timeout_s)
        if r.status_code in (404, 405, 501):
            raise RouteIndisponible(route)
        r.raise_for_status()
        return r.json()

    async def rechercher(self, requete: str, user_id: str) -> ResultatRag:
        """/chunks si disponible, sinon /ask. L'absence de /chunks est mémorisée puis retestée périodiquement."""
        if self._chunks_a_tester():
            try:
                payload = await self._post(CFG.rag_route_chunks, requete, user_id)
                self._chunks_ko_depuis = None
                return normaliser(payload, requete, "chunks")
            except RouteIndisponible:
                self._chunks_ko_depuis = time.monotonic()
                if CFG.rag_mode == "chunks":
                    raise
        payload = await self._post(CFG.rag_route_ask, requete, user_id)
        return normaliser(payload, requete, "ask")
