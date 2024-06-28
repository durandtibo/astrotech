r"""Contain the implementation of a simple model which is composed of 3 modules: network, criterion and metrics."""

from __future__ import annotations

__all__ = ["Model"]

from typing import TYPE_CHECKING

from karbonn.utils import setup_module

from astrotech.model.base import BaseModel
from astrotech.model.criteria import NoLoss

if TYPE_CHECKING:
    from torch.nn import Module


class Model(BaseModel):
    r"""Implement a simple model which is composed of 3 modules:
    network, criterion and metrics.

    Args:
        network: The network module or its configuration.
        criterion: The criterion module or its configuration.
            ``None`` means no criterion is used so the model cannot
            be trained because no loss is computed.

    Example usage:

    ```pycon

    >>> import torch
    >>> from astrotech.model import Model
    >>> from astrotech.model.network import Network
    >>> from astrotech.model.criteria import Loss
    >>> model = Model(
    ...     network=Network(
    ...         module=torch.nn.Linear(4, 6), input_keys=["input"], output_keys=["prediction"]
    ...     ),
    ...     criterion=Loss(criterion=torch.nn.MSELoss()),
    ... )
    >>> model
    Model(
      (network): Network(
        (input_keys): ('input',)
        (output_keys): ('prediction',)
        (module): Linear(in_features=4, out_features=6, bias=True)
      )
      (criterion): Loss(
        (prediction): prediction
        (target): target
        (weight): 1.0
        (criterion): MSELoss()
      )
    )
    >>> out = model({"input": torch.randn(2, 4), "target": torch.randn(2, 6)})
    >>> out
    {'prediction': tensor([[...]], grad_fn=<AddmmBackward0>), 'loss': tensor(..., grad_fn=<MulBackward0>)}

    ```
    """

    def __init__(
        self,
        network: Module | dict,
        criterion: Module | dict | None = None,
    ) -> None:
        super().__init__()
        self.network = setup_module(network)
        self.criterion = setup_module(criterion or NoLoss())

    def forward(self, batch: dict) -> dict:
        net_out = self.network(batch)
        cri_out = self.criterion(net_out, batch) if self.criterion else {}
        return net_out | cri_out
