import asyncio
import json

from websockets.asyncio.client import connect

from server.app import running_server
from server.protocol import parse_message
from server.session import Session
from sim.env import NeurolapEnv


def test_parse_valid_and_clipped_input():
    m = parse_message('{"type":"input","steering":5,"throttle":-9}')
    assert m == {"type": "input", "steering": 1.0, "throttle": -1.0}


def test_parse_rejects_invalid():
    invalid = [
        "nope",
        "[]",
        '{"type":"x"}',
        '{"type":"input","steering":"a","throttle":0}',
        '{"type":"input","steering":NaN,"throttle":0}',
        '{"type":"input","steering":true,"throttle":0}',
        '{"type":"reset","seed":"7"}',
    ]
    for raw in invalid:
        assert parse_message(raw) is None


def test_parse_reset():
    assert parse_message('{"type":"reset","seed":7}') == {"type": "reset", "seed": 7}
    assert parse_message('{"type":"reset"}') == {"type": "reset", "seed": None}


def test_tick_applies_latest_input():
    s = Session()
    s.set_input(0.0, 1.0)
    s.set_input(0.0, 0.5)
    msgs = s.tick()
    assert len(msgs) == 1 and msgs[0]["type"] == "state" and msgs[0]["step"] == 1
    assert msgs[0]["car"]["speed"] > 0


def test_same_seed_same_track_message():
    s = Session()
    a = s.reset(seed=5)[0]
    b = s.reset(seed=5)[0]
    assert a == b and a["type"] == "track" and a["seed"] == 5


def test_new_episode_after_end_with_new_seed():
    s = Session(env=NeurolapEnv(max_steps=3), end_pause_ticks=2, seed=0)
    first_seed = s.snapshot()[0]["seed"]
    for _ in range(3):
        last = s.tick()[0]
    assert last["done"] and last["outcome"] == "timeout"
    assert s.tick()[0]["type"] == "state"  # pause : état figé
    msgs = s.tick()  # fin de pause : nouveau circuit
    assert msgs[0]["type"] == "track" and msgs[0]["seed"] != first_seed
    assert msgs[1]["step"] == 0 and not msgs[1]["done"]


def test_websocket_integration():
    async def scenario() -> None:
        async with running_server(Session(), port=0, fps=60) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                view = json.loads(await ws.recv())
                assert view == {"type": "view", "view": "play"}
                first = json.loads(await ws.recv())
                assert first["type"] == "track" and len(first["center"]) > 100
                await ws.send(json.dumps({"type": "input", "steering": 0, "throttle": 1}))
                for _ in range(100):
                    msg = json.loads(await ws.recv())
                    if msg["type"] == "state" and msg["car"]["speed"] > 1.0:
                        return
                raise AssertionError("la voiture n'a pas accéléré")

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_steering_ramps_up_then_recenters():
    s = Session()
    s.set_input(1.0, 0.0)
    prev, values = 0.0, []
    for _ in range(12):
        s.tick()
        values.append(s.steering)
        assert s.steering >= prev
        prev = s.steering
    assert 0.0 < values[0] < 0.5 and values[-1] == 1.0  # montée progressive
    s.set_input(0.0, 0.0)
    ticks = 0
    while s.steering > 0.0:
        s.tick()
        ticks += 1
    assert ticks <= 5  # recentrage plus rapide que la montée


def test_parse_mode():
    assert parse_message('{"type":"mode","mode":"pilot"}') == {"type": "mode", "mode": "pilot"}
    assert parse_message('{"type":"mode","mode":"autre"}') is None


def test_pilot_mode_drives_and_ignores_keyboard():
    s = Session()
    s.set_mode("pilot")
    s.set_input(-1.0, -1.0)  # le clavier ne doit pas compter en mode pilote
    last = None
    for _ in range(60):
        last = s.tick()[0]
    assert last is not None and last["mode"] == "pilot"
    assert last["car"]["speed"] > 1.0  # le pilote a accéléré malgré le clavier en marche arrière
