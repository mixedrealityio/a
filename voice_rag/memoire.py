"""Mémoire locale par utilisateur : 10 dernières réponses RAG (avec menus) + derniers tours de conversation."""
import asyncio
import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .config import CFG
from .rag_client import ResultatRag

STOP = set("""le la les un une des du de au aux et ou en dans sur pour par avec sans que qui quoi est sont
ce cet cette ces mon ma mes ton ta tes son sa ses notre votre leur plus moins tres comment quel quelle quels
quelles pourquoi the of and to for is are what how""".split())

ORDINAUX = {"premier": 1, "premiere": 1, "1": 1, "deuxieme": 2, "second": 2, "seconde": 2, "deux": 2, "2": 2,
            "troisieme": 3, "trois": 3, "3": 3, "quatrieme": 4, "quatre": 4, "4": 4, "cinquieme": 5, "cinq": 5,
            "5": 5, "sixieme": 6, "six": 6, "6": 6, "dernier": -1, "derniere": -1}
ORDINAUX_COURTS = {"un": 1, "une": 1}      # ambigus : acceptés seulement dans une phrase de 3 mots au plus


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def mots(s: str) -> set[str]:
    return {m for m in norm(s).split() if len(m) >= 3 and m not in STOP}


def similarite(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ma, mb = mots(a), mots(b)
    jaccard = len(ma & mb) / len(ma | mb) if ma and mb else 0.0
    return max(SequenceMatcher(None, na, nb).ratio(), jaccard)


def choix_menu(texte: str, menu: list[str]) -> str | None:
    """Retourne le libellé de l'option choisie (« la deuxième », « option 2 », ou le libellé lui-même)."""
    if not texte or not menu:
        return None
    ms = norm(texte).split()
    if 0 < len(ms) <= 6:
        for m in ms:
            n = ORDINAUX.get(m) or (ORDINAUX_COURTS.get(m) if len(ms) <= 3 else None)
            if n == -1:
                return menu[-1]
            if n and n <= len(menu):
                return menu[n - 1]
    meilleur = max(menu, key=lambda o: similarite(texte, o))
    if similarite(texte, meilleur) >= 0.72:
        return meilleur
    mt = mots(texte)
    for o in menu:
        mo = mots(o)
        if mo and mo <= mt:
            return o
    return None


class Memoire:
    def __init__(self, dossier: str | None = None):
        self.dossier = Path(dossier or CFG.mem_dir)
        self._cache: dict[str, dict] = {}

    def _fichier(self, user: str) -> Path:
        return self.dossier / (re.sub(r"[^A-Za-z0-9_.-]", "_", user)[:80] or "anonyme") / "memoire.json"

    def _etat(self, user: str) -> dict:
        if user not in self._cache:
            try:
                self._cache[user] = json.loads(self._fichier(user).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache[user] = {"rag": [], "tours": []}
        return self._cache[user]

    # ------------------------------------------------------------------ écriture
    def ajouter_rag(self, user: str, res: ResultatRag):
        if not res.extraits and not res.menu:
            return
        e = self._etat(user)
        e["rag"] = [r for r in e["rag"] if norm(r["requete"]) != norm(res.requete)]
        e["rag"].append(res.to_dict())
        del e["rag"][:-CFG.mem_taille]

    def ajouter_tour(self, user: str, question: str, reponse: str):
        e = self._etat(user)
        e["tours"].append({"ts": time.time(), "question": question, "reponse": reponse})
        del e["tours"][:-max(CFG.mem_tours, 1) * 4]

    async def sauver(self, user: str):
        etat, chemin = json.dumps(self._etat(user), ensure_ascii=False), self._fichier(user)

        def _ecrire():
            chemin.parent.mkdir(parents=True, exist_ok=True)
            tmp = chemin.with_suffix(".tmp")
            tmp.write_text(etat, encoding="utf-8")
            os.replace(tmp, chemin)
        await asyncio.to_thread(_ecrire)

    # ------------------------------------------------------------------ lecture
    def historique(self, user: str) -> list[dict]:
        return self._etat(user)["tours"][-CFG.mem_tours:]

    def dernier_menu(self, user: str) -> list[str]:
        for r in reversed(self._etat(user)["rag"]):
            if r.get("menu"):
                return r["menu"]
        return []

    def chercher(self, user: str, requete: str, transcription: str = "") -> tuple[ResultatRag | None, str]:
        """Cherche d'abord en local. Retourne (résultat ou None, requête effective).

        1. Si l'utilisateur choisit une option du dernier menu, la requête devient le libellé de l'option.
        2. Si une des 10 dernières réponses porte (presque) la même requête, elle est réutilisée.
        3. Pour un choix de menu, les extraits des 10 dernières réponses qui couvrent l'option sont réutilisés.
        Sinon (None, requête) : l'orchestrateur interroge le backend.
        """
        entrees = [ResultatRag.from_dict(r) for r in self._etat(user)["rag"]]
        option = choix_menu(transcription, self.dernier_menu(user)) or choix_menu(requete, self.dernier_menu(user))
        req = option or requete
        frais = [r for r in entrees if time.time() - r.ts <= CFG.mem_ttl_s]
        if frais:
            meilleur = max(frais, key=lambda r: similarite(req, r.requete))
            if meilleur.extraits and similarite(req, meilleur.requete) >= CFG.mem_seuil_requete:
                meilleur.route = "memoire"
                return meilleur, req
        if option and mots(option):
            cible, notes = mots(option), []
            for r in frais:
                for ex in r.extraits:
                    couverture = len(cible & mots(ex.texte)) / len(cible)
                    if couverture >= CFG.mem_seuil_menu:
                        notes.append((couverture, ex))
            if notes:
                notes.sort(key=lambda n: -n[0])
                uniques = list({ex.id: ex for _, ex in notes}.values())[:CFG.rag_top_k]
                return ResultatRag(requete=req, extraits=uniques, route="memoire"), req
        return None, req
