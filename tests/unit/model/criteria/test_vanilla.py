from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from objectory import OBJECT_TARGET
from torch.nn import CrossEntropyLoss, L1Loss, Module, MSELoss

from astrotech import constants as ct
from astrotech.model.criteria import Loss

SIZES = (1, 2)


##########################
#     Tests for Loss     #
##########################


@pytest.mark.parametrize(
    ("criterion", "criterion_cls"),
    [
        (MSELoss(), MSELoss),
        (CrossEntropyLoss(), CrossEntropyLoss),
        ({OBJECT_TARGET: "torch.nn.MSELoss"}, MSELoss),
    ],
)
def test_loss_criterion(criterion: dict | Module, criterion_cls: type[Module]) -> None:
    assert isinstance(Loss(criterion).criterion, criterion_cls)


@pytest.mark.parametrize("device", get_available_devices())
def test_loss_mse_correct(device: str) -> None:
    device = torch.device(device)
    criterion = Loss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_loss_mse_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = Loss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(1.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_loss_l1_correct(device: str) -> None:
    device = torch.device(device)
    criterion = Loss(L1Loss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_loss_l1_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = Loss(L1Loss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(1.0, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_loss_cross_entropy(device: str) -> None:
    device = torch.device(device)
    criterion = Loss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, dtype=torch.long, device=device)},
        ),
        {ct.LOSS: torch.tensor(1.0986122886681098, device=device)},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prediction_key", ["my_prediction", "output"])
@pytest.mark.parametrize("target_key", ["my_target", "target"])
def test_loss_mse_custom_keys(device: str, prediction_key: str, target_key: str) -> None:
    device = torch.device(device)
    criterion = Loss(MSELoss(), prediction_key=prediction_key, target_key=target_key).to(
        device=device
    )
    assert objects_are_equal(
        criterion(
            {prediction_key: torch.ones(2, 3, device=device)},
            {target_key: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )
