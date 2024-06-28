from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices
from objectory import OBJECT_TARGET
from torch.nn import L1Loss, Module, MSELoss

from astrotech import constants as ct
from astrotech.model.criteria import Loss, ModeLoss

SIZES = (1, 2)


##############################
#     Tests for ModeLoss     #
##############################


def test_mode_loss_repr() -> None:
    assert repr(ModeLoss(train=Loss(MSELoss()), eval=Loss(L1Loss()))).startswith("ModeLoss(")


def test_mode_loss_str() -> None:
    assert str(ModeLoss(train=Loss(MSELoss()), eval=Loss(L1Loss()))).startswith("ModeLoss(")


@pytest.mark.parametrize(
    "criterion",
    [
        Loss(MSELoss()),
        {
            OBJECT_TARGET: "astrotech.model.criteria.Loss",
            "criterion": {OBJECT_TARGET: "torch.nn.MSELoss"},
        },
    ],
)
def test_mode_loss_criterion(criterion: dict | Module) -> None:
    criterion = ModeLoss(train=criterion, eval=criterion)
    assert isinstance(criterion.train_criterion, Loss)
    assert isinstance(criterion.eval_criterion, Loss)


@pytest.mark.parametrize("device", get_available_devices())
def test_mode_loss_forward_train(device: str) -> None:
    device = torch.device(device)
    criterion = ModeLoss(train=Loss(MSELoss()), eval=Loss(L1Loss())).to(device=device)
    criterion.train()
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: 2 * torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.zeros(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(4.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_mode_loss_forward_eval(device: str) -> None:
    device = torch.device(device)
    criterion = ModeLoss(train=Loss(MSELoss()), eval=Loss(L1Loss())).to(device=device)
    criterion.eval()
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: 2 * torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.zeros(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(2.0, device=device)},
    )
