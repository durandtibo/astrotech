from __future__ import annotations

import pytest
from coola import objects_are_equal
from minrecord import MinScalarRecord, Record
from torch import nn

from astrotech.engine.state import EngineState
from astrotech.utils.asset import AssetNotFoundError

NAMES = ("name1", "name2")


#################################
#     Tests for EngineState     #
#################################


def test_engine_state_str() -> None:
    assert str(EngineState()).startswith("EngineState(")


def test_engine_state_default_values() -> None:
    state = EngineState()
    assert state.epoch == -1
    assert state.iteration == -1
    assert state.max_epochs == 1
    assert state.random_seed == 9984043075503325450
    assert len(state._records) == 0
    assert len(state._modules) == 0


@pytest.mark.parametrize("epoch", [1, 2])
def test_engine_state_epoch(epoch: int) -> None:
    assert EngineState(epoch=epoch).epoch == epoch


@pytest.mark.parametrize("iteration", [1, 2])
def test_engine_state_iteration(iteration: int) -> None:
    assert EngineState(iteration=iteration).iteration == iteration


@pytest.mark.parametrize("max_epochs", [1, 2])
def test_engine_state_max_epochs(max_epochs: int) -> None:
    assert EngineState(max_epochs=max_epochs).max_epochs == max_epochs


@pytest.mark.parametrize("random_seed", [1, 2])
def test_engine_state_random_seed(random_seed: int) -> None:
    assert EngineState(random_seed=random_seed).random_seed == random_seed


@pytest.mark.parametrize("key", NAMES)
def test_engine_state_add_record_with_key(key: str) -> None:
    state = EngineState()
    state.add_record(MinScalarRecord("loss"), key)
    assert state._records.has_record(key)


@pytest.mark.parametrize("key", NAMES)
def test_engine_state_add_record_without_key(key: str) -> None:
    state = EngineState()
    state.add_record(MinScalarRecord(key))
    assert state._records.has_record(key)


@pytest.mark.parametrize("name", NAMES)
def test_engine_state_add_module(name: str) -> None:
    state = EngineState()
    state.add_module(name, nn.Linear(4, 5))
    assert state._modules.has_asset(name)


def test_engine_state_get_best_values_without_record() -> None:
    assert EngineState().get_best_values() == {}


def test_engine_state_get_best_values_with_record() -> None:
    state = EngineState()
    record = MinScalarRecord("loss")
    state.add_record(record)
    record.add_value(1.2, step=0)
    record.add_value(0.8, step=1)
    assert state.get_best_values() == {"loss": 0.8}


def test_engine_state_get_record_exists() -> None:
    state = EngineState()
    record = MinScalarRecord("loss")
    state.add_record(record)
    assert state.get_record("loss") is record


def test_engine_state_get_record_does_not_exist() -> None:
    state = EngineState()
    record = state.get_record("loss")
    assert isinstance(record, Record)
    assert len(record) == 0


def test_engine_state_get_records() -> None:
    manager = EngineState()
    record1 = MinScalarRecord("loss")
    record2 = MinScalarRecord("accuracy")
    manager.add_record(record1)
    manager.add_record(record2)
    assert manager.get_records() == {"loss": record1, "accuracy": record2}


def test_engine_state_get_records_empty() -> None:
    assert EngineState().get_records() == {}


def test_engine_state_get_module() -> None:
    state = EngineState()
    state.add_module("my_module", nn.Linear(4, 5))
    assert isinstance(state.get_module("my_module"), nn.Linear)


def test_engine_state_get_module_missing() -> None:
    state = EngineState()
    with pytest.raises(AssetNotFoundError, match="The asset 'my_module' does not exist"):
        state.get_module("my_module")


def test_engine_state_has_record_true() -> None:
    state = EngineState()
    state.add_record(MinScalarRecord("loss"))
    assert state.has_record("loss")


def test_engine_state_has_record_false() -> None:
    assert not EngineState().has_record("loss")


def test_engine_state_has_module_true() -> None:
    state = EngineState()
    state.add_module("my_module", nn.Linear(4, 5))
    assert state.has_module("my_module")


def test_engine_state_has_module_false() -> None:
    assert not EngineState().has_module("my_module")


def test_engine_state_increment_epoch_1() -> None:
    state = EngineState()
    assert state.epoch == -1
    state.increment_epoch()
    assert state.epoch == 0


def test_engine_state_increment_epoch_2() -> None:
    state = EngineState()
    assert state.epoch == -1
    state.increment_epoch(2)
    assert state.epoch == 1


def test_engine_state_increment_iteration_1() -> None:
    state = EngineState()
    assert state.iteration == -1
    state.increment_iteration()
    assert state.iteration == 0


def test_engine_state_increment_iteration_2() -> None:
    state = EngineState()
    assert state.iteration == -1
    state.increment_iteration(2)
    assert state.iteration == 1


def test_engine_state_load_state_dict() -> None:
    state = EngineState()
    state.load_state_dict(
        {
            "epoch": 5,
            "iteration": 101,
            "records": {},
            "modules": {},
        }
    )
    assert state.epoch == 5
    assert state.iteration == 101


def test_engine_state_state_dict_load_state_dict() -> None:
    state = EngineState()
    state.increment_epoch(6)
    state_dict = state.state_dict()
    state.increment_epoch(2)
    state.load_state_dict(state_dict)
    assert state.epoch == 5


def test_engine_state_remove_module_exists() -> None:
    state = EngineState()
    state.add_module("my_module", nn.Linear(4, 5))
    assert state.has_module("my_module")
    state.remove_module("my_module")
    assert not state.has_module("my_module")


def test_engine_state_remove_module_does_not_exist() -> None:
    state = EngineState()
    with pytest.raises(
        AssetNotFoundError,
        match="The asset 'my_module' does not exist so it is not possible to remove it",
    ):
        state.remove_module("my_module")


def test_engine_state_state_dict() -> None:
    assert objects_are_equal(
        EngineState().state_dict(),
        {"epoch": -1, "iteration": -1, "records": {}, "modules": {}},
    )
