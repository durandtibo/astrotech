from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from astrotech import constants as ct
from astrotech.model.criteria import Loss, SumLoss

SIZES = (1, 2)


#############################
#     Tests for SumLoss     #
#############################


def test_sum_loss_repr() -> None:
    assert repr(
        SumLoss({"mse": Loss(torch.nn.MSELoss()), "l1": Loss(torch.nn.L1Loss())})
    ).startswith("SumLoss(")


def test_sum_loss_str() -> None:
    assert str(
        SumLoss({"mse": Loss(torch.nn.MSELoss()), "l1": Loss(torch.nn.L1Loss())})
    ).startswith("SumLoss(")


def test_sum_loss_init() -> None:
    criterion = SumLoss({"mse": torch.nn.MSELoss(), "l1": torch.nn.L1Loss()})
    assert isinstance(criterion["mse"], torch.nn.MSELoss)
    assert isinstance(criterion["l1"], torch.nn.L1Loss)


@pytest.mark.parametrize("device", get_available_devices())
def test_sum_loss_forward_1(device: str) -> None:
    device = torch.device(device)
    criterion = SumLoss({"mse": Loss(torch.nn.MSELoss())}).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: 2 * torch.ones(2, 3, device=device)},
        ),
        {
            ct.LOSS: torch.tensor(4.0, device=device),
            f"{ct.LOSS}_mse": torch.tensor(4.0, device=device),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_sum_loss_forward_2(device: str) -> None:
    device = torch.device(device)
    criterion = SumLoss({"mse": Loss(torch.nn.MSELoss()), "l1": Loss(torch.nn.L1Loss())}).to(
        device=device
    )
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: 2 * torch.ones(2, 3, device=device)},
        ),
        {
            ct.LOSS: torch.tensor(6.0, device=device),
            f"{ct.LOSS}_mse": torch.tensor(4.0, device=device),
            f"{ct.LOSS}_l1": torch.tensor(2.0, device=device),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_sum_loss_forward_3(device: str) -> None:
    device = torch.device(device)
    criterion = SumLoss(
        {
            "mse": Loss(torch.nn.MSELoss()),
            "l1": Loss(torch.nn.L1Loss()),
            "sl1": Loss(torch.nn.SmoothL1Loss()),
        }
    ).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: 2 * torch.ones(2, 3, device=device)},
        ),
        {
            ct.LOSS: torch.tensor(7.5, device=device),
            f"{ct.LOSS}_mse": torch.tensor(4.0, device=device),
            f"{ct.LOSS}_l1": torch.tensor(2.0, device=device),
            f"{ct.LOSS}_sl1": torch.tensor(1.5, device=device),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_sum_loss_forward_correct(device: str) -> None:
    device = torch.device(device)
    criterion = SumLoss({"mse": Loss(torch.nn.MSELoss()), "l1": Loss(torch.nn.L1Loss())}).to(
        device=device
    )
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {
            ct.LOSS: torch.tensor(0.0, device=device),
            f"{ct.LOSS}_mse": torch.tensor(0.0, device=device),
            f"{ct.LOSS}_l1": torch.tensor(0.0, device=device),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_sum_loss_forward_incorrect(device: str) -> None:
    device = torch.device(device)
    criterion = SumLoss({"mse": Loss(torch.nn.MSELoss()), "l1": Loss(torch.nn.L1Loss())}).to(
        device=device
    )
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: 2 * torch.ones(2, 3, device=device)},
        ),
        {
            ct.LOSS: torch.tensor(6.0, device=device),
            f"{ct.LOSS}_mse": torch.tensor(4.0, device=device),
            f"{ct.LOSS}_l1": torch.tensor(2.0, device=device),
        },
    )
