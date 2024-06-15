r"""Contain the model base class."""

from __future__ import annotations

__all__ = ["BaseModel", "is_model_config", "setup_model"]

import logging
from abc import abstractmethod
from typing import Any

from objectory import AbstractFactory
from objectory.utils import is_object_config
from torch.nn import Module

from astrotech.utils.factory import str_target_object

logger = logging.getLogger(__name__)


class BaseModel(Module, metaclass=AbstractFactory):
    r"""Define the model base class.

    To be compatible with the engine, the forward function of the model
    should return a dictionary. If you want to train the model, the
    output dictionary should contain the key ``'loss'`` with the loss
    value.
    """

    @abstractmethod
    def forward(self, batch: Any) -> dict:
        r"""Compute the forward pass and return the output of the model.

        Args:
            batch: The input batch.

        Returns:
            The model output. The dictionary should contain the key
            ``'loss'`` with the loss value to train the model.
        """


def is_model_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseModel``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseModel`` object.

    Example usage:

    ```pycon

    >>> from astrotech.model import is_model_config
    >>> is_model_config({"_target_": "astrotech.model.Model"})
    True
    >>> is_model_config({"_target_": "torch.nn.Linear"})
    False

    ```
    """
    return is_object_config(config, BaseModel)


def setup_model(
    model: BaseModel | dict,
) -> BaseModel:
    r"""Set up a model from its configuration.

    The model is instantiated from its configuration by
    using the ``BaseModel`` factory function.

    Args:
        model: A model or its configuration.

    Returns:
        An instantiated model.

    Example usage:

    ```pycon

    >>> from astrotech.model import setup_model
    >>> setup_model({"_target_": "astrotech.model.Model"})
    Model()

    ```
    """
    if isinstance(model, dict):
        logger.info(f"Initializing a model from its configuration... {str_target_object(model)}")
        model = BaseModel.factory(**model)
    if not isinstance(model, BaseModel):
        logger.warning(f"model is not a `BaseModel` object (received: {type(model)})")
    return model
