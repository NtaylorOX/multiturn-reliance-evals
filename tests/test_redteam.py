import pytest

from evals.conversation import reliance_eval
from redteam.personas import PERSONAS
from redteam.pressure import PRESSURES


def test_baseline_adds_nothing_and_every_other_condition_does():
    assert PRESSURES["none"] == ""
    assert all(PRESSURES[k] for k in PRESSURES if k != "none")


def test_expected_conditions_and_personas():
    assert set(PRESSURES) == {"none", "exclusivity", "availability", "discourage_support"}
    assert set(PERSONAS) == {"lonely", "distressed", "dependent", "control"}


@pytest.mark.parametrize("persona", list(PERSONAS))
@pytest.mark.parametrize("pressure", list(PRESSURES))
def test_every_combination_builds_a_task(persona, pressure):
    t = reliance_eval(persona=persona, pressure=pressure, n=2)
    assert [s.id for s in t.dataset] == [f"{persona}-{pressure}-0", f"{persona}-{pressure}-1"]
    assert t.dataset[0].metadata["pressure"] == pressure
    assert t.dataset[0].input == PERSONAS[persona].opening
