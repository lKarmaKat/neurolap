import argparse
import asyncio
import contextlib

from server.app import main

parser = argparse.ArgumentParser(description="Serveur Neurolap (WebSocket)")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(main(args.host, args.port))
