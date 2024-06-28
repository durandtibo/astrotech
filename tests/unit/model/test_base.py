from __future__ import annotations

import logging

import torch
from objectory import OBJECT_TARGET
from torch.nn import Linear

from astrotech.model import Model, is_model_config, setup_model
from astrotech.model.criteria import Loss
from astrotech.model.network import Network
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

#####################################
#     Tests for is_model_config     #
#####################################


def test_is_model_config_true() -> None:
    assert is_model_config({OBJECT_TARGET: "astrotech.model.Model"})


def test_is_model_config_false() -> None:
    assert not is_model_config({OBJECT_TARGET: "torch.nn.Linear"})


#################################
#     Tests for setup_model     #
#################################


def test_setup_model_object() -> None:
    model = Model(
        network=Network(
            module=torch.nn.Linear(4, 6), input_keys=["input"], output_keys=["prediction"]
        ),
        criterion=Loss(criterion=torch.nn.MSELoss()),
    )
    assert setup_model(model) is model


def test_setup_model_dict() -> None:
    assert isinstance(
        setup_model(
            {
                OBJECT_TARGET: "astrotech.model.Model",
                "network": {
                    OBJECT_TARGET: "astrotech.model.network.Network",
                    "module": {
                        OBJECT_TARGET: "torch.nn.Linear",
                        "in_features": 4,
                        "out_features": 6,
                    },
                    "input_keys": ["input"],
                    "output_keys": ["prediction"],
                },
                "criterion": {
                    OBJECT_TARGET: "astrotech.model.criteria.Loss",
                    "criterion": {OBJECT_TARGET: "torch.nn.MSELoss"},
                },
            }
        ),
        Model,
    )


def test_setup_model_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_model({OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}),
            Linear,
        )
        assert caplog.messages
