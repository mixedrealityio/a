"""Client PC : micro -> orchestrateur (WebSocket) -> haut-parleurs. Sera remplacé par l'app HoloLens.

    python -m voice_rag.client_micro --user alice                 # micro, détection de parole côté serveur
    python -m voice_rag.client_micro --appui                      # Entrée pour commencer / finir de parler
    python -m voice_rag.client_micro --camera 0                   # joint les dernières images de la webcam
    python -m voice_rag.client_micro --sans-micro                 # questions tapées au clavier

Commandes au clavier : /image chemin.jpg   /menu 2   /stop   /quit   ou une question en texte.
"""
import argparse
import asyncio
import base64
import json
import mimetypes
import queue
import sys
import threading
import time

import websockets

HZ_MICRO, TRAME = 16000, 640          # trames de 40 ms


class Lecteur:
    """Joue le PCM reçu dans un fil séparé ; `vider()` coupe la lecture (interruption)."""

    def __init__(self, hz: int):
        import sounddevice as sd
        self.file: queue.Queue = queue.Queue()
        self.fin_lecture = 0.0
        self.flux = sd.RawOutputStream(samplerate=hz, channels=1, dtype="int16")
        self.flux.start()
        threading.Thread(target=self._boucle, daemon=True).start()

    def _boucle(self):
        while True:
            pcm = self.file.get()
            self.flux.write(pcm)
            self.fin_lecture = time.monotonic()

    def jouer(self, pcm: bytes):
        self.file.put(pcm)

    def vider(self):
        with self.file.mutex:
            self.file.queue.clear()

    def actif(self) -> bool:
        return not self.file.empty() or time.monotonic() - self.fin_lecture < 0.4


async def envoyer_micro(ws, args, lecteur, etat):
    import sounddevice as sd
    boucle, file = asyncio.get_running_loop(), asyncio.Queue()
    flux = sd.RawInputStream(samplerate=HZ_MICRO, channels=1, dtype="int16", blocksize=TRAME,
                             callback=lambda data, *_: boucle.call_soon_threadsafe(file.put_nowait, bytes(data)))
    with flux:
        while True:
            pcm = await file.get()
            if args.appui and not etat["enregistre"]:
                continue
            if not args.interruption and lecteur and lecteur.actif():
                continue                      # micro coupé pendant la réponse (sinon : casque + --interruption)
            await ws.send(pcm)


async def envoyer_camera(ws, index: int, ips: float):
    import cv2
    cam = cv2.VideoCapture(index)
    try:
        while True:
            ok, image = await asyncio.to_thread(cam.read)
            if ok:
                h, l = image.shape[:2]
                image = cv2.resize(image, (640, int(h * 640 / l)))
                ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    await ws.send(json.dumps({"type": "video.frame", "data": base64.b64encode(jpg).decode()}))
            await asyncio.sleep(1 / ips)
    finally:
        cam.release()


async def lire_clavier(ws, args, etat):
    while True:
        ligne = (await asyncio.to_thread(sys.stdin.readline))
        if not ligne:
            return
        ligne = ligne.strip()
        if ligne == "/quit":
            return
        if ligne == "/stop":
            await ws.send(json.dumps({"type": "interrupt"}))
        elif ligne.startswith("/image "):
            chemin = ligne[7:].strip()
            with open(chemin, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = mimetypes.guess_type(chemin)[0] or "image/jpeg"
            await ws.send(json.dumps({"type": "image", "mime": mime, "data": data}))
            print("[image jointe à la prochaine question]")
        elif ligne.startswith("/menu "):
            await ws.send(json.dumps({"type": "menu.select", "index": int(ligne[6:])}))
        elif ligne:
            await ws.send(json.dumps({"type": "text", "question": ligne}))
        elif args.appui:
            etat["enregistre"] = not etat["enregistre"]
            print("[parlez… Entrée pour finir]" if etat["enregistre"] else "[envoyé]")
            if not etat["enregistre"]:
                await ws.send(json.dumps({"type": "turn.end"}))


async def recevoir(ws, lecteur):
    async for msg in ws:
        if isinstance(msg, bytes):
            if lecteur:
                lecteur.jouer(msg)
            continue
        e = json.loads(msg)
        t = e["type"]
        if t == "transcript":
            print(f"\nVous      : {e['text']}")
        elif t == "rag":
            print(f"[recherche {e['route']} : {e['nb_extraits']} extrait(s), {e['ms']} ms]")
        elif t == "answer.delta":
            print(f"Assistant : {e['text']}")
        elif t == "answer.card":
            for i, m in enumerate(e.get("menu") or [], 1):
                print(f"   {i}. {m}")
            sources = ", ".join(sorted({s['source'] for s in e.get("sources") or [] if s.get("source")}))
            if sources:
                print(f"   Sources : {sources}")
            if not e["controle"]["ok"]:
                print(f"   Contrôle : {e['controle']['problemes']}")
        elif t == "interrupt" and lecteur:
            lecteur.vider()
        elif t == "metrics":
            print(f"[{e}]")
        elif t == "error":
            print(f"ERREUR : {e['message']}")


async def principal(args):
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.start", "user_id": args.user, "sample_rate": HZ_MICRO,
                                  "vad": not args.appui, "barge_in": args.interruption}))
        pret = json.loads(await ws.recv())
        lecteur = None if args.sans_son else Lecteur(pret.get("sample_rate_out", 24000))
        etat = {"enregistre": False}
        taches = [asyncio.create_task(recevoir(ws, lecteur)), asyncio.create_task(lire_clavier(ws, args, etat))]
        if not args.sans_micro:
            taches.append(asyncio.create_task(envoyer_micro(ws, args, lecteur, etat)))
        if args.camera is not None:
            taches.append(asyncio.create_task(envoyer_camera(ws, args.camera, args.ips)))
        if args.image:
            with open(args.image, "rb") as f:
                await ws.send(json.dumps({"type": "image", "mime": mimetypes.guess_type(args.image)[0] or "image/jpeg",
                                          "data": base64.b64encode(f.read()).decode()}))
        print("Connecté. Parlez, ou tapez une question. /quit pour sortir.")
        _, en_cours = await asyncio.wait(taches, return_when=asyncio.FIRST_COMPLETED)
        for t in en_cours:
            t.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Client micro PC de l'orchestrateur vocal RAG")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--user", default="demo")
    p.add_argument("--appui", action="store_true", help="appuyer sur Entrée pour parler (pas de détection de parole)")
    p.add_argument("--interruption", action="store_true", help="couper la parole à l'assistant (casque conseillé)")
    p.add_argument("--image", help="image jointe à la première question")
    p.add_argument("--camera", type=int, help="index de la webcam dont les images sont jointes aux questions")
    p.add_argument("--ips", type=float, default=1.0, help="images par seconde envoyées depuis la webcam")
    p.add_argument("--sans-micro", action="store_true")
    p.add_argument("--sans-son", action="store_true")
    asyncio.run(principal(p.parse_args()))
