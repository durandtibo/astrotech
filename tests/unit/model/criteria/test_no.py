from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from astrotech.model.criteria import NoLoss

SIZES = (1, 2)


############################
#     Tests for NoLoss     #
############################


def test_no_loss_repr() -> None:
    assert repr(NoLoss()).startswith("NoLoss(")


def test_no_loss_str() -> None:
    assert str(NoLoss()).startswith("NoLoss(")


@pytest.mark.parametrize("device", get_available_devices())
def test_no_loss_mse_correct(device: str) -> None:
    device = torch.device(device)
    criterion = NoLoss().to(device=device)
    assert criterion({}, {}) == {}
