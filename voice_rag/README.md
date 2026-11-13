# Orchestrateur vocal RAG (Qwen-Omni)

Agent conversationnel vocal dont les réponses viennent uniquement du backend RAG.
Aujourd'hui le client est le micro d'un PC ; le protocole WebSocket est celui que l'app HoloLens utilisera.

```
micro PC / HoloLens ──WebSocket──► orchestrateur ──► Qwen-Omni (comprend, appelle l'outil, rédige en texte)
        ▲   audio + événements         │  ▲
        └──────────────────────────────┘  └──► backend RAG : /chunks, sinon /ask
```

## Un tour de parole

1. Le serveur détecte la fin de la phrase et envoie l'audio (plus image ou vidéo) à Qwen-Omni.
2. Qwen-Omni renvoie un appel de fonction `rechercher_rag(requete, transcription)`.
3. L'orchestrateur **intercepte** l'appel : mémoire locale d'abord, puis `/chunks`, sinon `/ask`.
4. Le résultat est **nettoyé**, puis renvoyé au modèle comme résultat d'outil.
5. Le modèle rédige en texte. Chaque phrase est **contrôlée**, puis lue à voix haute et envoyée en streaming.

## Fichiers

| Fichier | Rôle |
|---|---|
| `orchestrateur.py` | Serveur WebSocket `/ws`, détection de parole, pipeline d'un tour, `POST /texte` pour tester |
| `qwen.py` | Client Qwen-Omni en streaming : texte, appels d'outils (natifs ou balise `<tool_call>`), voix |
| `rag_client.py` | Routes `/chunks` et `/ask`, nettoyage des résultats |
| `controle.py` | Contrôle des réponses, découpage en phrases |
| `memoire.py` | Mémoire locale par utilisateur : 10 dernières réponses RAG, menus, historique |
| `prompts.py` | Prompt système et définition de l'outil |
| `client_micro.py` | Client PC : micro, haut-parleurs, image, webcam |
| `mock_backends.py` | Faux RAG et faux Qwen-Omni pour développer sans réseau |

## Démarrage

```bash
pip install -r voice_rag/requirements.txt
```

Essai sans rien d'autre (faux backends) :

```bash
uvicorn voice_rag.mock_backends:app --port 9000
```

```bash
uvicorn voice_rag.orchestrateur:app --port 8000
```

```bash
python -m voice_rag.client_micro --user alice
```

Avec les vrais services, avant de lancer l'orchestrateur :

```bash
export QWEN_BASE_URL="https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export QWEN_API_KEY="..."
export RAG_BASE_URL="https://votre-backend-sphere-xr"
```

## Backend RAG : `/chunks` ou `/ask`

`RAG_MODE=auto` (défaut) essaie `/chunks`. Si la route répond 404, 405 ou 501, l'orchestrateur passe à `/ask`
et reteste `/chunks` toutes les 60 secondes : le jour où `/chunks` existe, elle est utilisée sans rien changer.
`RAG_MODE=ask` ou `RAG_MODE=chunks` force une route.

Avec `/ask`, la réponse rédigée par le backend est jetée. Seuls `data` et `sources` sont gardés.
Si cette réponse contient une liste numérotée, elle est lue comme un menu avant d'être jetée.

Le corps envoyé est `{"question": ..., "top_k": ..., "user_id": ...}`. Le nom du champ se change avec
`RAG_CHAMP_QUESTION`. La lecture de la réponse accepte plusieurs formes (`chunks`, `results`, `data`, `documents`…) ;
si votre backend en utilise une autre, adaptez `normaliser()` dans `rag_client.py`.

## Contrôle des réponses

Une phrase est refusée si :

- elle contient un chiffre absent des extraits et de la question ;
- elle cite un repère `[S4]` qui n'existe pas ;
- moins de 45 % de ses mots importants se retrouvent dans les extraits (`ANCRAGE_SEUIL`) ;
- elle contient une formule de connaissance générale (« en général », « je pense »…).

Si le modèle répond sans avoir appelé l'outil avec un texte long ou chiffré, la recherche est forcée.
`CONTROLE_MODE=phrase` (défaut) remplace une phrase refusée par une phrase de repli et continue.
`CONTROLE_MODE=complet` attend toute la réponse, la fait corriger une fois, puis parle : plus sûr, plus lent.

Le contrôle d'ancrage est une heuristique sur les mots. Calibrez `ANCRAGE_SEUIL` sur vos documents.

## Mémoire locale

Un fichier par utilisateur dans `MEM_DIR` (`./memoire/<user>/memoire.json`) garde les 10 dernières réponses RAG.
Quand le RAG propose un menu et que l'utilisateur répond « la deuxième », « option 2 » ou le nom de l'option,
l'orchestrateur cherche d'abord dans ces 10 réponses et n'appelle le backend que s'il ne trouve rien.

## Image et vidéo

- Image : `/image chemin.jpg` dans le client, ou l'événement `image`. Elle est jointe à la question suivante.
- Vidéo : `--camera 0` envoie une image par seconde (`video.frame`). Les 6 dernières, vieilles de moins de
  8 secondes, sont jointes à chaque question. La HoloLens enverra les mêmes événements.

Qwen-Omni accepte du texte plus une seule autre modalité par message. Avec une question audio, l'image ou la vidéo
part donc dans un message utilisateur placé juste avant (`MEDIA_SEPARE=1`).

## Voix

`VOIX_MODE=omni` (défaut) fait lire chaque phrase validée par `QWEN_VOICE_MODEL`.
`VOIX_MODE=tts` utilise une route compatible OpenAI `/audio/speech` (`TTS_URL`).
`VOIX_MODE=aucune` n'envoie que le texte.

## Tests

```bash
python -m pytest voice_rag/tests -q
```

Les tests utilisent les faux backends. Le client micro, la webcam et les vrais services Qwen-Omni et RAG
ne sont pas couverts.
