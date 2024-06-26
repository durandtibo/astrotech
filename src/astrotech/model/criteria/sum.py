r"""Contain a criterion that computes the sum of multiple loss
functions."""

from __future__ import annotations

__all__ = ["SumLoss"]

from typing import TYPE_CHECKING

from karbonn.utils import setup_module
from torch.nn import Module, ModuleDict

from astrotech import constants as ct

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


class SumLoss(ModuleDict):
    r"""Implement a loss function that computes the sum of multiple loss
    functions.

    Args:
        criteria: The mapping of loss functions to sum. The keys are
            used to track the value of each loss function.

    Example usage:

    ```pycon

    >>> import torch
    >>> from astrotech.model.criteria import Loss, SumLoss
    >>> criterion = SumLoss({"mse": Loss(torch.nn.MSELoss()), "l1": Loss(torch.nn.L1Loss())})
    >>> criterion
    SumLoss(
      (mse): Loss(
        (prediction): prediction
        (target): target
        (weight): 1.0
        (criterion): MSELoss()
      )
      (l1): Loss(
        (prediction): prediction
        (target): target
        (weight): 1.0
        (criterion): L1Loss()
      )
    )
    >>> loss = criterion(
    ...     net_out={"prediction": torch.rand(2, 4)}, batch={"target": torch.rand(2, 4)}
    ... )
    >>> loss
    {'loss': tensor(...)}

    ```
    """

    def __init__(self, criteria: Mapping[str, Module | dict]) -> None:
        super().__init__({key: setup_module(criterion) for key, criterion in criteria.items()})

    def forward(self, net_out: dict, batch: dict) -> dict[str, Tensor]:
        r"""Return the loss value given the network output and the batch.

        Args:
            net_out: The network output which contains the prediction.
            batch: The batch which contains the target.

        Returns:
            A dict with the loss value.
        """
        out = {ct.LOSS: 0.0}
        for key, criterion in self.items():
            out[f"{ct.LOSS}_{key}"] = criterion(net_out, batch)[ct.LOSS]
            out[ct.LOSS] += out[f"{ct.LOSS}_{key}"]
        return out
