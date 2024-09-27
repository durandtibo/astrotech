from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola.utils.tensor import get_available_devices
from objectory import OBJECT_TARGET

from astrotech.model.network import Network

if TYPE_CHECKING:
    from torch.nn import Module

SIZES = (1, 2, 3)


@pytest.fixture
def network() -> Network:
    return Network(
        module=torch.nn.Linear(in_features=4, out_features=6),
        input_keys=["input"],
        output_keys=["output"],
    )


#############################
#     Tests for Network     #
#############################


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Linear(in_features=4, out_features=6),
        {OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6},
    ],
)
def test_network_init(module: Module | dict) -> None:
    network = Network(module=module, input_keys=["input"], output_keys=["output"])
    assert isinstance(network.module, torch.nn.Linear)


def test_network_repr(network: Network) -> None:
    assert repr(network).startswith("Network(")


def test_network_str(network: Network) -> None:
    assert str(network).startswith("Network(")


def test_network_input_keys(network: Network) -> None:
    assert network.input_keys == ("input",)


def test_network_output_keys(network: Network) -> None:
    assert network.output_keys == ("output",)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_network_forward_linear(network: Network, device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    network = network.to(device=device)
    network.train(mode)
    output = network({"input": torch.ones(batch_size, 4, device=device)})
    assert len(output) == 1
    assert output["output"].shape == (batch_size, 6)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_network_forward_bilinear(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    network = Network(
        module=torch.nn.Bilinear(in1_features=4, in2_features=5, out_features=6),
        input_keys=["input1", "input2"],
        output_keys=["output"],
    ).to(device=device)
    network.train(mode)
    output = network(
        {
            "input1": torch.ones(batch_size, 4, device=device),
            "input2": torch.ones(batch_size, 5, device=device),
        }
    )
    assert len(output) == 1
    assert output["output"].shape == (batch_size, 6)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_network_forward_gru(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    network = Network(
        module=torch.nn.GRU(input_size=4, hidden_size=6, batch_first=True),
        input_keys=["input"],
        output_keys=["output", "h_n"],
    ).to(device=device)
    network.train(mode)
    output = network({"input": torch.ones(batch_size, 8, 4, device=device)})
    assert len(output) == 2
    assert output["output"].shape == (batch_size, 8, 6)
    assert output["h_n"].shape == (1, batch_size, 6)
