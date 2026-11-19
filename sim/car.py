"""Voiture type vélo cinématique. Repère écran : y vers le bas, θ croissant = virage à droite."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CarParams:
    wheelbase: float = 2.5  # empattement L (m)
    max_steer: float = 0.5  # braquage max δ_max (rad)
    accel: float = 8.0  # accélération à plein gaz (m/s²)
    drag: float = 0.5  # traînée linéaire (1/s)
    max_speed: float = 20.0  # vitesse max avant (m/s)
    max_reverse_speed: float = 5.0  # vitesse max arrière (m/s)
    car_length: float = 4.0  # emprise pour les collisions (m)
    car_width: float = 1.8

    @property
    def max_yaw_rate(self) -> float:
        """Vitesse angulaire maximale atteignable (rad/s), pour normaliser l'observation."""
        return self.max_speed / self.wheelbase * math.tan(self.max_steer)


@dataclass(frozen=True)
class CarState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # cap θ (rad)
    speed: float = 0.0  # vitesse signée (m/s)
    yaw_rate: float = 0.0  # vitesse angulaire ω (rad/s)


def step_car(
    state: CarState, steering: float, throttle: float, params: CarParams, dt: float
) -> CarState:
    """Avance la voiture d'un pas dt (Euler explicite). Actions supposées déjà dans [-1, 1]."""
    delta = steering * params.max_steer
    speed = state.speed + (throttle * params.accel - params.drag * state.speed) * dt
    speed = min(max(speed, -params.max_reverse_speed), params.max_speed)
    yaw_rate = speed / params.wheelbase * math.tan(delta)
    return CarState(
        x=state.x + speed * math.cos(state.heading) * dt,
        y=state.y + speed * math.sin(state.heading) * dt,
        heading=state.heading + yaw_rate * dt,
        speed=speed,
        yaw_rate=yaw_rate,
    )


def car_corners(state: CarState, params: CarParams) -> list[tuple[float, float]]:
    """Les 4 coins du rectangle de la voiture (centré sur x, y)."""
    c, s = math.cos(state.heading), math.sin(state.heading)
    hl, hw = params.car_length / 2, params.car_width / 2
    return [
        (state.x + c * dx - s * dy, state.y + s * dx + c * dy)
        for dx, dy in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw))
    ]
