r"""Contain a criterion wrapper to make PyTorch criteria compatible with
``astrotech.model.Model``."""

from __future__ import annotations

__all__ = ["Loss"]

from typing import TYPE_CHECKING

from coola.utils import str_mapping
from karbonn.utils import setup_module
from torch.nn import Module

from astrotech import constants as ct

if TYPE_CHECKING:
    from torch import Tensor


class Loss(Module):
    r"""Implement a wrapper to make compatible most of the PyTorch loss
    functions (a.k.a. criterion) with ``astrotech.model.Model``.

    This wrapper assumes the loss function has two inputs:

        - a tensor of prediction.
        - a tensor of target

    The shape and type of the tensors depend on the loss function used.

    Args:
        criterion: The loss module (a.k.a. criterion) or its
            configuration.
        prediction_key: The key that indicates the prediction in
            ``net_out``.
        target_key (str): The key that indicates the target in
            ``batch``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from astrotech.model.criteria import Loss
    >>> # Initialization with a nn.Module
    >>> criterion = Loss(criterion=torch.nn.MSELoss())
    >>> criterion
    Loss(
      (prediction): prediction
      (target): target
      (weight): 1.0
      (criterion): MSELoss()
    )
    >>> # Initialization with a config
    >>> criterion = Loss(criterion={"_target_": "torch.nn.MSELoss"})
    >>> criterion
    Loss(
      (prediction): prediction
      (target): target
      (weight): 1.0
      (criterion): MSELoss()
    )
    >>> # Customize keys.
    >>> criterion = Loss(
    ...     criterion=torch.nn.MSELoss(),
    ...     prediction_key="my_prediction",
    ...     target_key="my_target",
    ... )
    >>> criterion
    Loss(
      (prediction): my_prediction
      (target): my_target
      (weight): 1.0
      (criterion): MSELoss()
    )
    >>> loss = criterion(
    ...     net_out={"my_prediction": torch.rand(2, 4)}, batch={"my_target": torch.rand(2, 4)}
    ... )
    >>> loss
    {'loss': tensor(...)}

    ```
    """

    def __init__(
        self,
        criterion: Module | dict,
        prediction_key: str = ct.PREDICTION,
        target_key: str = ct.TARGET,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.criterion = setup_module(criterion)
        self._prediction_key = prediction_key
        self._target_key = target_key
        self._weight = float(weight)

    def extra_repr(self) -> str:
        return str_mapping(
            {
                "prediction": self._prediction_key,
                "target": self._target_key,
                "weight": self._weight,
            }
        )

    def forward(self, net_out: dict, batch: dict) -> dict[str, Tensor]:
        r"""Return the loss value given the network output and the batch.

        Args:
            net_out: The network output which contains the prediction.
            batch: The batch which contains the target.

        Returns:
            A dict with the loss value.
        """
        return {
            ct.LOSS: self._weight
            * self.criterion(net_out[self._prediction_key], batch[self._target_key])
        }
