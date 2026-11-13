"""Client Qwen-Omni en streaming (API compatible OpenAI) : texte, appels d'outils, voix."""
import base64
import json
import re
from dataclasses import dataclass, field

import httpx

from .config import CFG
from .prompts import NOM_OUTIL, OUTIL_RAG, PROMPT_FORCER_OUTIL, PROMPT_LECTEUR, PROMPT_OUTIL_TEXTE

_BALISE = "<tool_call>"
_APPEL_TEXTE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)


class ErreurLLM(Exception):
    def __init__(self, code: int, corps: str):
        super().__init__(f"LLM HTTP {code}: {corps[:300]}")
        self.code = code


@dataclass
class AppelOutil:
    id: str
    nom: str
    args: dict = field(default_factory=dict)


def _json_tolerant(s: str) -> dict:
    try:
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {}
    except ValueError:
        m = re.search(r"\{.*\}", s or "", re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except ValueError:
            return {}


class FiltreAppelTexte:
    """Retient du flux de texte tout ce qui suit une balise <tool_call> (modèles sans appels natifs)."""

    def __init__(self):
        self.visible_en_attente, self.cache, self.dans_appel = "", "", False

    def ajouter(self, delta: str) -> str:
        if self.dans_appel:
            self.cache += delta
            return ""
        t = self.visible_en_attente + delta
        i = t.find(_BALISE)
        if i >= 0:
            self.dans_appel, self.cache, self.visible_en_attente = True, t[i:], ""
            return t[:i]
        garde = next((n for n in range(min(len(_BALISE) - 1, len(t)), 0, -1) if _BALISE.startswith(t[-n:])), 0)
        self.visible_en_attente = t[len(t) - garde:] if garde else ""
        return t[:len(t) - garde] if garde else t

    def fin(self) -> tuple[str, list[AppelOutil]]:
        appels = []
        for i, brut in enumerate(_APPEL_TEXTE.findall(self.cache)):
            d = _json_tolerant(brut)
            args = d.get("arguments") or d.get("parameters") or {}
            appels.append(AppelOutil(id=f"txt_{i}", nom=str(d.get("name", "")),
                                     args=args if isinstance(args, dict) else _json_tolerant(str(args))))
        reste, self.visible_en_attente = self.visible_en_attente, ""
        return reste, appels


def contenu_media(images=(), frames=()) -> list[dict]:
    parts = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}} for mime, b64 in images]
    urls = [f"data:image/jpeg;base64,{f}" for f in frames]
    if len(urls) >= 2 and CFG.video_style == "video":
        parts.append({"type": "video", "video": urls})          # vidéo = liste d'images (format DashScope)
    else:
        parts += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    return parts


def contenu_audio(wav: bytes) -> dict:
    b64 = base64.b64encode(wav).decode()
    if CFG.audio_input_style == "audio_url":
        return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}
    data = f"data:;base64,{b64}" if CFG.audio_input_style == "dashscope" else b64
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


class QwenClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.outils_natifs = CFG.tool_mode != "prompt"

    # ------------------------------------------------------------------ messages d'outils
    def prompt_systeme(self, base: str) -> str:
        return base if self.outils_natifs else base + PROMPT_OUTIL_TEXTE

    def msg_appels(self, appels: list[AppelOutil]) -> dict:
        if self.outils_natifs:
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": a.id, "type": "function",
                 "function": {"name": a.nom, "arguments": json.dumps(a.args, ensure_ascii=False)}} for a in appels]}
        corps = "".join(f'<tool_call>{json.dumps({"name": a.nom, "arguments": a.args}, ensure_ascii=False)}</tool_call>'
                        for a in appels)
        return {"role": "assistant", "content": corps}

    def msg_resultat(self, appel: AppelOutil, contenu: str) -> dict:
        if self.outils_natifs:
            return {"role": "tool", "tool_call_id": appel.id, "content": contenu}
        return {"role": "user", "content": f"<tool_response>\n{contenu}\n</tool_response>"}

    # ------------------------------------------------------------------ appel en streaming
    async def flux(self, messages: list[dict], *, outils: bool = False, forcer: bool = False,
                   audio: bool = False, modele: str | None = None):
        """Génère des événements ("texte", str), ("audio", bytes), ("appel", AppelOutil)."""
        corps = {"model": modele or CFG.qwen_model, "messages": messages, "stream": True,
                 "temperature": CFG.qwen_temperature, "max_tokens": CFG.qwen_max_tokens,
                 "modalities": ["text", "audio"] if audio else ["text"]}
        if audio:
            corps["audio"] = {"voice": CFG.qwen_voice, "format": "wav"}
        else:
            corps.update(CFG.qwen_extra)
        if outils and self.outils_natifs:
            corps["tools"] = [OUTIL_RAG]
            if forcer:
                corps["tool_choice"] = {"type": "function", "function": {"name": NOM_OUTIL}}
        elif forcer:
            corps["messages"] = messages + [{"role": "user", "content": PROMPT_FORCER_OUTIL}]
        entetes = {"Authorization": f"Bearer {CFG.qwen_api_key}"} if CFG.qwen_api_key else {}

        natifs: dict[int, dict] = {}
        filtre = FiltreAppelTexte()
        async with self.http.stream("POST", CFG.qwen_base_url + "/chat/completions", json=corps,
                                    headers=entetes, timeout=CFG.llm_timeout_s) as r:
            if r.status_code >= 400:
                raise ErreurLLM(r.status_code, (await r.aread()).decode(errors="replace"))
            async for ligne in r.aiter_lines():
                if not ligne.startswith("data:"):
                    continue
                data = ligne[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    choix = (json.loads(data).get("choices") or [{}])[0]
                except ValueError:
                    continue
                delta = choix.get("delta") or {}
                son = delta.get("audio") or {}
                if son.get("data"):
                    yield "audio", base64.b64decode(son["data"])
                texte = delta.get("content") if isinstance(delta.get("content"), str) else None
                texte = texte or son.get("transcript")
                if texte:
                    visible = filtre.ajouter(texte)
                    if visible:
                        yield "texte", visible
                for tc in delta.get("tool_calls") or []:
                    e = natifs.setdefault(tc.get("index", 0), {"id": "", "nom": "", "args": ""})
                    e["id"] = tc.get("id") or e["id"]
                    f = tc.get("function") or {}
                    e["nom"] = f.get("name") or e["nom"]
                    e["args"] += f.get("arguments") or ""
        reste, appels_texte = filtre.fin()
        if reste:
            yield "texte", reste
        for i in sorted(natifs):
            e = natifs[i]
            yield "appel", AppelOutil(id=e["id"] or f"call_{i}", nom=e["nom"], args=_json_tolerant(e["args"]))
        for a in appels_texte:
            yield "appel", a

    async def flux_avec_repli(self, messages, **kw):
        """Comme `flux`, mais bascule en appels d'outils textuels si le serveur refuse `tools` (TOOL_MODE=auto)."""
        try:
            async for evt in self.flux(messages, **kw):
                yield evt
        except ErreurLLM as e:
            if not (e.code in (400, 422) and kw.get("outils") and self.outils_natifs and CFG.tool_mode == "auto"):
                raise
            self.outils_natifs = False
            messages = [dict(messages[0], content=messages[0]["content"] + PROMPT_OUTIL_TEXTE)] + messages[1:]
            async for evt in self.flux(messages, **kw):
                yield evt

    # ------------------------------------------------------------------ voix
    async def dire(self, texte: str):
        """Génère le PCM 16 bits mono (CFG.audio_out_hz) du texte validé, par blocs, au fil de la synthèse."""
        if CFG.voix_mode == "aucune":
            return
        reste = b""
        if CFG.voix_mode == "tts":
            corps = {"model": CFG.tts_model, "input": texte, "voice": CFG.qwen_voice, "response_format": "pcm"}
            entetes = {"Authorization": f"Bearer {CFG.qwen_api_key}"} if CFG.qwen_api_key else {}
            async with self.http.stream("POST", CFG.tts_url, json=corps, headers=entetes,
                                        timeout=CFG.llm_timeout_s) as r:
                if r.status_code >= 400:
                    raise ErreurLLM(r.status_code, (await r.aread()).decode(errors="replace"))
                async for bloc in r.aiter_bytes():
                    bloc = reste + bloc
                    reste = bloc[len(bloc) & ~1:]
                    if len(bloc) > 1:
                        yield bloc[:len(bloc) & ~1]
            return
        messages = [{"role": "system", "content": PROMPT_LECTEUR},
                    {"role": "user", "content": f"<lire>{texte}</lire>"}]
        premier = True
        async for type_, val in self.flux(messages, audio=True, modele=CFG.qwen_voice_model):
            if type_ != "audio":
                continue
            if premier and val[:4] == b"RIFF":
                val = val[44:]                      # en-tête WAV éventuel
            premier = False
            bloc = reste + val
            reste = bloc[len(bloc) & ~1:]
            if len(bloc) > 1:
                yield bloc[:len(bloc) & ~1]
