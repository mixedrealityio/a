"""Configuration, lue dans les variables d'environnement (voir README)."""
import json
import os


def _s(cle, defaut):
    return os.environ.get(cle, defaut)


class Config:
    def __init__(self):
        # --- Qwen-Omni (API compatible OpenAI) ---
        self.qwen_base_url = _s("QWEN_BASE_URL", _s("DASHSCOPE_BASE_URL", "http://127.0.0.1:9000/v1")).rstrip("/")
        self.qwen_api_key = _s("QWEN_API_KEY", _s("DASHSCOPE_API_KEY", ""))
        self.qwen_model = _s("QWEN_MODEL", "qwen3.8-omni-flash")            # comprend + appelle l'outil + rédige (texte)
        self.qwen_voice_model = _s("QWEN_VOICE_MODEL", "qwen3.5-omni-flash")  # lit la réponse validée (audio)
        self.qwen_voice = _s("QWEN_VOICE", "Tina")
        self.qwen_extra = json.loads(_s("QWEN_EXTRA_JSON", '{"reasoning_effort": "none"}'))
        self.qwen_temperature = float(_s("QWEN_TEMPERATURE", "0"))
        self.qwen_max_tokens = int(_s("QWEN_MAX_TOKENS", "300"))
        self.audio_input_style = _s("AUDIO_INPUT_STYLE", "dashscope")  # dashscope | openai | audio_url
        self.video_style = _s("VIDEO_STYLE", "video")                  # video (liste d'images) | images
        self.media_separe = _s("MEDIA_SEPARE", "1") == "1"   # 1 seule modalité non-texte par message utilisateur
        self.tool_mode = _s("TOOL_MODE", "auto")             # auto | native | prompt
        self.llm_timeout_s = float(_s("LLM_TIMEOUT_S", "30"))

        # --- Backend RAG ---
        self.rag_base_url = _s("RAG_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
        self.rag_api_key = _s("RAG_API_KEY", "")
        self.rag_mode = _s("RAG_MODE", "auto")               # auto (/chunks puis /ask) | chunks | ask
        self.rag_route_chunks = _s("RAG_ROUTE_CHUNKS", "/chunks")
        self.rag_route_ask = _s("RAG_ROUTE_ASK", "/ask")
        self.rag_champ_question = _s("RAG_CHAMP_QUESTION", "question")
        self.rag_top_k = int(_s("RAG_TOP_K", "4"))
        self.rag_seuil = float(_s("RAG_SEUIL", "0.0"))       # score minimal (ignoré si le backend ne donne pas de score)
        self.rag_timeout_s = float(_s("RAG_TIMEOUT_S", "8"))
        self.rag_recheck_s = float(_s("RAG_RECHECK_S", "60"))  # délai avant de retester /chunks
        self.rag_max_car_extrait = int(_s("RAG_MAX_CAR_EXTRAIT", "700"))
        self.rag_max_car_total = int(_s("RAG_MAX_CAR_TOTAL", "2400"))
        self.max_recherches = int(_s("MAX_RECHERCHES", "2"))

        # --- Mémoire locale ---
        self.mem_dir = _s("MEM_DIR", "./memoire")
        self.mem_taille = int(_s("MEM_TAILLE", "10"))        # dernières réponses RAG gardées par utilisateur
        self.mem_tours = int(_s("MEM_TOURS", "6"))           # tours de conversation renvoyés au modèle
        self.mem_ttl_s = float(_s("MEM_TTL_S", "1800"))
        self.mem_seuil_requete = float(_s("MEM_SEUIL_REQUETE", "0.86"))
        self.mem_seuil_menu = float(_s("MEM_SEUIL_MENU", "0.6"))

        # --- Contrôle des réponses ---
        self.controle_mode = _s("CONTROLE_MODE", "phrase")   # phrase (rapide) | complet (plus sûr, plus lent)
        self.ancrage_seuil = float(_s("ANCRAGE_SEUIL", "0.45"))
        self.ancrage_min_mots = int(_s("ANCRAGE_MIN_MOTS", "4"))
        self.sans_rag_max_car = int(_s("SANS_RAG_MAX_CAR", "220"))

        # --- Voix ---
        self.voix_mode = _s("VOIX_MODE", "omni")             # omni | tts | aucune
        self.tts_url = _s("TTS_URL", "")                     # route compatible OpenAI /audio/speech
        self.tts_model = _s("TTS_MODEL", "tts-1")
        self.audio_out_hz = int(_s("AUDIO_OUT_HZ", "24000"))
        self.voix_paralleles = int(_s("VOIX_PARALLELES", "2"))

        # --- Détection de parole (côté serveur) ---
        self.vad_seuil = float(_s("VAD_SEUIL", "500"))       # RMS sur du PCM 16 bits
        self.vad_debut_ms = int(_s("VAD_DEBUT_MS", "120"))
        self.vad_fin_ms = int(_s("VAD_FIN_MS", "700"))
        self.vad_preroll_ms = int(_s("VAD_PREROLL_MS", "300"))
        self.vad_max_s = int(_s("VAD_MAX_S", "30"))

        # --- Image / vidéo ---
        self.video_max_frames = int(_s("VIDEO_MAX_FRAMES", "6"))
        self.video_fraicheur_s = float(_s("VIDEO_FRAICHEUR_S", "8"))


CFG = Config()
