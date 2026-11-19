import json

import numpy as np
import pytest

from agents.evolution import Episode, evolve, run_episode
from agents.mlp import MLPPilot
from agents.runs import RunRecorder, RunStore
from server.hub import Hub
from server.protocol import parse_message
from server.replay import RESTART_PAUSE_TICKS, ReplaySession
from server.session import Session


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    """Un petit entraînement enregistré : 3 générations de 6 individus."""
    root = tmp_path_factory.mktemp("runs")
    rec = RunRecorder(root / "essai")
    evolve(
        generations=3,
        pop_size=6,
        n_elite=2,
        seed=1,
        workers=0,
        seeds_per_gen=2,
        on_generation=lambda g: rec.record(g, {"fitness": 0.5, "finished": 1.0, "lap_s": 20.0}),
    )
    return root


def test_recorder_and_store_roundtrip(runs_dir):
    store = RunStore(runs_dir)
    assert store.runs() == [{"name": "essai", "generations": 3}]
    pop, fit, seeds = store.load_generation("essai", 1)
    assert pop.shape[0] == 6 and len(fit) == 6 and len(seeds) == 2
    assert np.all(np.diff(fit) <= 0)  # triée par fitness décroissante
    hist = store.history("essai")
    assert [h["generation"] for h in hist] == [0.0, 1.0, 2.0]
    assert hist[0]["val_fitness"] == 0.5


def test_store_rejects_unsafe_names(runs_dir):
    store = RunStore(runs_dir)
    for bad in ("../x", "a/b", "..", ""):
        with pytest.raises(ValueError):
            store.n_generations(bad)


def test_episode_matches_run_episode():
    pilot = MLPPilot(np.random.default_rng(3).normal(0, 0.5, 98))
    ep = Episode(pilot, 7)
    while not ep.done:
        ep.step()
    assert ep.result() == run_episode(pilot, 7)


def test_replay_select_and_clamp(runs_dir):
    r = ReplaySession(RunStore(runs_dir))
    msgs = r.select()  # défaut : dernier entraînement, dernière génération
    kinds = [m["type"] for m in msgs]
    assert kinds == ["runs", "history", "track", "replay_state"]
    assert msgs[-1]["generation"] == 2 and len(msgs[-1]["cars"]) == 6
    assert r.select(generation=99)[-1]["generation"] == 2
    assert r.select(generation=0, circuit=1)[-1]["circuit"] == 1
    assert r.select(circuit=9)[-1]["circuit"] == 1  # circuit borné (2 circuits : 0 et 1)


def test_replay_runs_to_the_end_then_loops(runs_dir):
    r = ReplaySession(RunStore(runs_dir))
    r.select(generation=0, speed=8)
    for _ in range(2000):
        state = r.tick()[-1]
        if state["running"] == 0:
            break
    assert state["running"] == 0 and state["step"] > 0
    for _ in range(RESTART_PAUSE_TICKS + 1):
        state = r.tick()[-1]
    assert state["step"] < 50  # relancée


def test_replay_is_deterministic(runs_dir):
    def run() -> list:
        r = ReplaySession(RunStore(runs_dir))
        r.select(generation=1, circuit=0)
        return [r.tick()[-1]["cars"] for _ in range(40)]

    assert run() == run()


def test_replay_without_runs(tmp_path):
    r = ReplaySession(RunStore(tmp_path / "vide"))
    msgs = r.select()
    assert [m["type"] for m in msgs] == ["runs", "history"] and msgs[0]["runs"] == []
    assert r.tick() == []


def test_parse_view_and_replay_messages():
    assert parse_message('{"type":"view","view":"replay"}') == {"type": "view", "view": "replay"}
    assert parse_message('{"type":"view","view":"x"}') is None
    ok = parse_message('{"type":"replay","run":"essai","generation":3,"speed":4}')
    assert ok == {"type": "replay", "run": "essai", "generation": 3, "speed": 4}
    for bad in (
        '{"type":"replay","run":"../etc"}',
        '{"type":"replay","generation":-1}',
        '{"type":"replay","generation":1.5}',
        '{"type":"replay","speed":3}',
    ):
        assert parse_message(bad) is None


def test_hub_switches_views(runs_dir):
    hub = Hub(Session(), RunStore(runs_dir))
    assert hub.snapshot()[0] == {"type": "view", "view": "play"}
    msgs = hub.handle({"type": "view", "view": "replay"})
    assert msgs[0]["view"] == "replay" and "replay_state" in [m["type"] for m in msgs]
    assert hub.tick()[-1]["type"] == "replay_state"
    back = hub.handle({"type": "reset", "seed": 4})
    assert back[0]["view"] == "play" and hub.tick()[0]["type"] == "state"
    json.dumps(msgs)  # tous les messages sont sérialisables
