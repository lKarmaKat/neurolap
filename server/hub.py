"""Hub : aiguille les messages entre le jeu (Session) et la rediffusion (ReplaySession)."""

from __future__ import annotations

from typing import Any

from agents.runs import RunStore
from server.replay import ReplaySession
from server.session import Session


class Hub:
    def __init__(self, session: Session, store: RunStore | None = None) -> None:
        self.session = session
        self.replay = ReplaySession(store or RunStore())
        self.view = "play"  # "play" (jeu) ou "replay" (rediffusion)

    def _view_msg(self) -> dict[str, Any]:
        return {"type": "view", "view": self.view}

    def snapshot(self) -> list[dict[str, Any]]:
        body = self.replay.snapshot() if self.view == "replay" else self.session.snapshot()
        return [self._view_msg(), *body]

    def tick(self) -> list[dict[str, Any]]:
        return self.replay.tick() if self.view == "replay" else self.session.tick()

    def handle(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Traite un message client valide ; retourne les messages à diffuser."""
        kind = msg["type"]
        if kind == "input":
            self.session.set_input(msg["steering"], msg["throttle"])
        elif kind == "mode":
            self.session.set_mode(msg["mode"])
        elif kind == "reset":
            self.view = "play"
            return [self._view_msg(), *self.session.reset(msg["seed"])]
        elif kind == "view":
            self.view = msg["view"]
            if self.view == "replay" and not self.replay.run:
                self.replay.select()
            return self.snapshot()
        elif kind == "replay":
            self.view = "replay"
            sel = {k: msg[k] for k in ("run", "generation", "circuit", "speed") if k in msg}
            return [self._view_msg(), *self.replay.select(**sel)]
        return []
