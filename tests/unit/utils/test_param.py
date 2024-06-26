from __future__ import annotations

import pytest
import torch
from torch.nn import Linear, Parameter, UninitializedParameter

from astrotech.utils.param import is_parameter, is_uninitialized_parameter

##################################
#     Tests for is_parameter     #
##################################


@pytest.mark.parametrize("value", [Parameter(torch.ones(2, 3)), UninitializedParameter()])
def test_is_parameter_true(value: Parameter) -> None:
    assert is_parameter(value)


@pytest.mark.parametrize("value", [torch.ones(2, 3), Linear(4, 6), "abc"])
def test_is_parameter_false(value: Parameter) -> None:
    assert not is_parameter(value)


################################################
#     Tests for is_uninitialized_parameter     #
################################################


def test_is_uninitialized_parameter_true() -> None:
    assert is_uninitialized_parameter(UninitializedParameter())


@pytest.mark.parametrize(
    "value", [Parameter(torch.ones(2, 3)), torch.ones(2, 3), Linear(4, 6), "abc"]
)
def test_is_uninitialized_parameter_false(value: Parameter) -> None:
    assert not is_uninitialized_parameter(value)
