from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from astrotech import constants as ct
from astrotech.model import Model
from astrotech.model.criteria import Loss
from astrotech.model.network import Network

SIZES = (1, 2, 3)


@pytest.fixture()
def model() -> Model:
    return Model(
        network=Network(
            module=torch.nn.Linear(in_features=4, out_features=6),
            input_keys=["input"],
            output_keys=["prediction"],
        ),
        criterion=Loss(criterion=torch.nn.MSELoss()),
    )


###########################
#     Tests for Model     #
###########################


def test_model_repr(model: Model) -> None:
    assert repr(model).startswith("Model(")


def test_model_str(model: Model) -> None:
    assert str(model).startswith("Model(")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_model_forward(model: Model, device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    model = model.to(device=device)
    model.train(mode)
    output = model(
        {
            "input": torch.ones(batch_size, 4, device=device),
            "target": torch.ones(batch_size, 6, device=device),
        }
    )
    assert len(output) == 2
    assert output["prediction"].shape == (batch_size, 6)
    assert isinstance(output[ct.LOSS].item(), float)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_model_forward_no_criterion(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    model = Model(
        network=Network(
            module=torch.nn.Linear(in_features=4, out_features=6),
            input_keys=["input"],
            output_keys=["prediction"],
        ),
    ).to(device=device)
    model.train(mode)
    output = model(
        {
            "input": torch.ones(batch_size, 4, device=device),
            "target": torch.ones(batch_size, 6, device=device),
        }
    )
    assert len(output) == 1
    assert output["prediction"].shape == (batch_size, 6)
