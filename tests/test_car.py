from sim.car import CarParams, CarState, step_car

P = CarParams()
DT = 1 / 30


def run(steering: float, throttle: float, n: int, state: CarState | None = None) -> CarState:
    s = state or CarState()
    for _ in range(n):
        s = step_car(s, steering, throttle, P, DT)
    return s


def test_straight_keeps_heading():
    s = run(0.0, 1.0, 60)
    assert s.heading == 0.0 and s.y == 0.0 and s.x > 0


def test_steer_right_increases_heading():
    assert run(1.0, 1.0, 30).heading > 0
    assert run(-1.0, 1.0, 30).heading < 0


def test_speed_capped_forward_and_reverse():
    assert run(0, 1.0, 1000).speed <= P.max_speed
    assert run(0, -1.0, 1000).speed >= -P.max_reverse_speed


def test_drag_slows_without_throttle():
    fast = run(0, 1.0, 60)
    assert run(0, 0.0, 60, fast).speed < fast.speed


def test_no_turn_when_stopped():
    assert run(1.0, 0.0, 30).heading == 0.0
