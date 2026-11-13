"""Contrôle des réponses avant qu'elles soient prononcées. Du code, pas un LLM : rapide et déterministe."""
import re
import unicodedata
from dataclasses import dataclass, field

from .config import CFG

REPERE = re.compile(r"\[S(\d+)\]")
FORMULES_INTERDITES = [
    r"\ben tant qu['’ ]?(ia|intelligence artificielle|mod[eè]le)",
    r"\bmes connaissances\b", r"\bà ma connaissance\b", r"\ben g[ée]n[ée]ral\b", r"\bg[ée]n[ée]ralement\b",
    r"\bil est probable\b", r"\bprobablement\b", r"\bje (pense|crois|suppose)\b",
]
# Mots de liaison de plus de 3 lettres, ignorés pour le calcul d'ancrage.
STOP_ANCRAGE = set("""avec pour dans cette votre notre vous nous plus moins tout tous toute toutes etre avoir
faire fait faut sont peut pouvez peuvent doit devez voici selon aussi donc mais comme ainsi alors puis
ensuite avant apres entre sans sous chez leur leurs elle elles ceci cela celui celle documentation
information indique indiquent choisir option options souhaitez voulez quel quelle""".split())

_GROUPE = r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?!\d)"
_SIMPLE = r"\d+(?:[.,]\d+)*"
_NOMBRE = re.compile(f"{_GROUPE}|{_SIMPLE}")
_NOMBRE_SIMPLE = re.compile(_SIMPLE)
_FIN_PHRASE = re.compile(r"[.!?…]+(?:\s*\[S\d+\])*[»\"”)]*(?=\s|$)")


def _valeur(s: str) -> float | None:
    s = re.sub(r"[   ]", "", s)
    if "," in s and "." in s:      # 1.234,56 ou 1,234.56
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif s.count(",") > 1 or s.count(".") > 1:
        s = s.replace(",", "").replace(".", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def nombres(texte: str, large: bool = False) -> set[float]:
    """Nombres d'un texte. `large` ajoute la lecture sans regroupement des milliers (pour les sources)."""
    texte = REPERE.sub(" ", texte)
    trouves = _NOMBRE.findall(texte) + (_NOMBRE_SIMPLE.findall(texte) if large else [])
    return {v for n in trouves if (v := _valeur(n)) is not None}


def racines(texte: str) -> set[str]:
    t = unicodedata.normalize("NFKD", REPERE.sub(" ", texte).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {m[:6] for m in re.findall(r"[a-z0-9]+", t) if len(m) >= 4 and m not in STOP_ANCRAGE}


@dataclass
class ContexteControle:
    question: str = ""
    sources: str = ""              # extraits + libellés de sources + menu
    nb_extraits: int = 0
    _nombres: set = field(default_factory=set)
    _racines: set = field(default_factory=set)

    def __post_init__(self):
        self._nombres = nombres(self.sources, large=True) | nombres(self.question, large=True) | set(map(float, range(11)))
        self._racines = racines(self.sources) | racines(self.question)


def verifier_phrase(phrase: str, ctx: ContexteControle) -> list[str]:
    """Retourne la liste des problèmes d'une phrase (liste vide = phrase acceptée)."""
    if "je n'ai pas cette information" in phrase.lower().replace("’", "'"):
        return []
    problemes = []
    for n in REPERE.findall(phrase):
        if not 1 <= int(n) <= ctx.nb_extraits:
            problemes.append(f"Le repère [S{n}] n'existe pas.")
    inventes = nombres(phrase) - ctx._nombres
    if inventes:
        liste = ", ".join(f"{v:g}" for v in sorted(inventes))
        problemes.append(f"Chiffres absents de la documentation : {liste}.")
    for motif in FORMULES_INTERDITES:
        if re.search(motif, phrase, re.I):
            problemes.append("Formulation qui s'appuie sur des connaissances générales.")
            break
    r = racines(phrase)
    if ctx.nb_extraits == 0:
        if len(phrase) > CFG.sans_rag_max_car:
            problemes.append("Réponse longue sans recherche documentaire.")
    elif len(r) >= CFG.ancrage_min_mots:
        ancrage = len(r & ctx._racines) / len(r)
        if ancrage < CFG.ancrage_seuil:
            problemes.append(f"Phrase peu appuyée sur la documentation (ancrage {ancrage:.0%}).")
    return problemes


def nettoyer_pour_voix(phrase: str) -> str:
    """Retire les repères [S1] et le markdown : ce texte est affiché et prononcé."""
    p = REPERE.sub("", phrase)
    p = re.sub(r"\*\*|__|`+|^\s*[-•*#]+\s*", "", p)
    p = re.sub(r"\s+([.,;:!?])", r"\1", p.replace(" ;", ";"))
    return re.sub(r"\s+", " ", p).strip()


def reperes_utilises(texte: str) -> list[int]:
    return sorted({int(n) for n in REPERE.findall(texte)})


class Decoupeur:
    """Découpe un flux de texte en phrases complètes, au fil de l'eau."""

    def __init__(self, min_car: int = 12):
        self.tampon, self.min_car = "", min_car

    def ajouter(self, delta: str) -> list[str]:
        self.tampon += delta
        phrases, debut = [], 0
        for m in _FIN_PHRASE.finditer(self.tampon):
            if m.end() == len(self.tampon):      # la suite peut encore arriver (ex. « 3.5 », « [S1] »)
                break
            if m.end() - debut >= self.min_car:
                phrases.append(self.tampon[debut:m.end()].strip())
                debut = m.end()
        self.tampon = self.tampon[debut:]
        return phrases

    def vider(self) -> list[str]:
        reste, self.tampon = self.tampon.strip(), ""
        return [reste] if reste else []


def decouper(texte: str) -> list[str]:
    d = Decoupeur()
    return d.ajouter(texte) + d.vider()
