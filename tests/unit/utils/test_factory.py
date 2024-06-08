from __future__ import annotations

import pytest
from objectory import OBJECT_TARGET
from torch.nn import Module, ReLU

from astrotech.utils.factory import setup_object, str_target_object

##################################
#     Tests for setup_object     #
##################################


@pytest.mark.parametrize("module", [ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_object(module: Module | dict) -> None:
    assert isinstance(setup_object(module), ReLU)


def test_setup_object_object() -> None:
    module = ReLU()
    assert setup_object(module) is module


#######################################
#     Tests for str_target_object     #
#######################################


def test_str_target_object_with_target() -> None:
    assert str_target_object({OBJECT_TARGET: "something.MyClass"}) == "something.MyClass"


def test_str_target_object_without_target() -> None:
    assert str_target_object({}) == "N/A"
