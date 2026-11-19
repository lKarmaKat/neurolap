import argparse
import asyncio
import contextlib

from agents.mlp import MLPPilot
from agents.runs import RunStore
from server.app import main

parser = argparse.ArgumentParser(description="Serveur Neurolap (WebSocket)")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--pilot", help="pilote entraîné (.npz) pour la touche P ; défaut : RayPilot")
parser.add_argument("--runs", default="models/runs", help="dossier des entraînements enregistrés")
args = parser.parse_args()
pilot = MLPPilot.load(args.pilot) if args.pilot else None
with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(main(args.host, args.port, pilot, RunStore(args.runs)))
