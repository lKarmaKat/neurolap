import math

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


def test_lateral_accel_never_exceeds_limit():
    for v in (2.0, 5.0, 10.0, 20.0, 34.0):
        s = CarState(speed=v)
        nxt = step_car(s, 1.0, 0.0, P, DT)
        assert abs(nxt.speed * nxt.yaw_rate) <= P.max_lateral_accel * 1.001


def test_full_steer_range_at_low_speed():
    s = step_car(CarState(speed=3.0), 1.0, 0.0, P, DT)
    expected = s.speed / P.wheelbase * math.tan(P.max_steer)
    assert math.isclose(s.yaw_rate, expected)


def test_steer_limit_decreases_with_speed():
    assert P.steer_limit(5.0) >= P.steer_limit(15.0) > P.steer_limit(30.0) > 0


def test_max_yaw_rate_bounds_all_speeds():
    for v in range(1, 36):
        nxt = step_car(CarState(speed=float(v)), 1.0, 0.0, P, DT)
        assert abs(nxt.yaw_rate) <= P.max_yaw_rate * 1.001
