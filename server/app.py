"""Serveur WebSocket : diffuse l'état de la session à pas fixe et reçoit les entrées clavier."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from websockets import broadcast
from websockets.asyncio.server import Server, ServerConnection, serve

from server.protocol import dumps, parse_message
from server.session import Session

FPS = 30  # cadence d'affichage ; le dt de la simulation reste celui de l'env


@contextlib.asynccontextmanager
async def running_server(
    session: Session, host: str = "127.0.0.1", port: int = 8765, fps: float = FPS
) -> AsyncIterator[Server]:
    clients: set[ServerConnection] = set()

    def send_all(msgs: list[dict[str, Any]]) -> None:
        for m in msgs:
            broadcast(clients, dumps(m))

    async def handler(ws: ServerConnection) -> None:
        clients.add(ws)
        try:
            for m in session.snapshot():
                await ws.send(dumps(m))
            async for raw in ws:
                msg = parse_message(raw)
                if msg is None:
                    continue
                if msg["type"] == "input":
                    session.set_input(msg["steering"], msg["throttle"])
                else:
                    send_all(session.reset(msg["seed"]))
        finally:
            clients.discard(ws)

    async def tick_loop() -> None:
        loop = asyncio.get_running_loop()
        period = 1.0 / fps
        nxt = loop.time()
        while True:
            send_all(session.tick())
            nxt += period  # échéance absolue : pas de dérive cumulée
            delay = nxt - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                nxt = loop.time()  # en retard : on ne rattrape pas en rafale
                await asyncio.sleep(0)

    async with serve(handler, host, port) as server:
        task = asyncio.create_task(tick_loop())
        try:
            yield server
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    async with running_server(Session(), host, port):
        print(f"Neurolap : ws://{host}:{port}")
        await asyncio.Future()  # tourne jusqu'à Ctrl+C
