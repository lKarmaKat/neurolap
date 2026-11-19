"""Protocole JSON client/serveur : encodage des messages sortants, validation des entrants."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from sim.env import NeurolapEnv
from sim.sensors import MAX_RANGE, RAY_ANGLES


def _r(a: Any) -> Any:
    """Arrondit à 2 décimales (allège les messages)."""
    return np.round(np.asarray(a, dtype=float), 2).tolist()


def encode_track(env: NeurolapEnv, seed: int) -> dict[str, Any]:
    t = env.track
    return {
        "type": "track",
        "seed": seed,
        "width": t.width,
        "length": round(t.length, 2),
        "center": _r(t.center),
        "left": _r(t.left),
        "right": _r(t.right),
        "ray_angles": _r(RAY_ANGLES),
        "ray_max_range": MAX_RANGE,
    }


def encode_state(
    env: NeurolapEnv, reward: float, terminated: bool, truncated: bool, info: dict[str, Any]
) -> dict[str, Any]:
    c = env.car
    if info["collision"]:
        outcome = "collision"
    elif terminated:
        outcome = "finished"
    elif truncated:
        outcome = "timeout"
    else:
        outcome = "running"
    return {
        "type": "state",
        "step": env.steps,
        "car": {
            "x": round(c.x, 3),
            "y": round(c.y, 3),
            "heading": round(c.heading, 4),
            "speed": round(c.speed, 3),
            "yaw_rate": round(c.yaw_rate, 4),
        },
        "rays": _r(info["ray_distances"]),
        "reward": round(reward, 6),
        "completion": round(info["completion"], 4),
        "outcome": outcome,
        "done": outcome != "running",
    }


def dumps(msg: dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"))


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        return None
    return float(v)


def _clip(v: float) -> float:
    return max(-1.0, min(1.0, v))


def parse_message(raw: str | bytes) -> dict[str, Any] | None:
    """Valide un message entrant. Retourne None s'il est invalide (il est alors ignoré)."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(msg, dict):
        return None
    if msg.get("type") == "input":
        steering, throttle = _num(msg.get("steering")), _num(msg.get("throttle"))
        if steering is None or throttle is None:
            return None
        return {"type": "input", "steering": _clip(steering), "throttle": _clip(throttle)}
    if msg.get("type") == "reset":
        seed = msg.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            return None
        return {"type": "reset", "seed": seed}
    return None
