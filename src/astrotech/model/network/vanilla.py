r"""Contain the implementation of a simple network."""

from __future__ import annotations

__all__ = ["Network"]


from typing import TYPE_CHECKING

import torch
from coola.utils import str_mapping
from karbonn.utils import setup_module
from torch.nn import Module

if TYPE_CHECKING:
    from collections.abc import Sequence


class Network(Module):
    r"""Implement a simple network module to adapt some
    ``torch.nn.Module``s to work with input and output dictionaries.

    Args:
        module: The wrapped module or its configuration.
        input_keys: The keys that are used to find the inputs of the
            wrapped module. The order of the keys should be the same
            that the order in the inputs of the forward function of
            the wrapped module.
        output_keys: The keys that are used to generate the outputs of
            the network. The order of the keys should be the same
            that the order in the outputs of the forward function of
            the wrapped module.

    ```pycon.

    >>> import torch
    >>> from astrotech.model.network import Network
    >>> network = Network(module=torch.nn.Linear(4, 6), input_keys=['input'], output_keys=['output'])
    >>> network
    Network(
      (input_keys): ('input',)
      (output_keys): ('output',)
      (module): Linear(in_features=4, out_features=6, bias=True)
    )
    >>> out = network({'input': torch.randn(2, 4)})
    >>> out
    {'output': tensor([[...]], grad_fn=<AddmmBackward0>)}
    >>> network = Network(module=torch.nn.Bilinear(4, 5, 6), input_keys=['input1', 'input2'], output_keys=['output'])
    >>> network
    Network(
      (input_keys): ('input1', 'input2')
      (output_keys): ('output',)
      (module): Bilinear(in1_features=4, in2_features=5, out_features=6, bias=True)
    )
    >>> out = network({'input1': torch.randn(2, 4), 'input2': torch.randn(2, 5)})
    >>> out
    {'output': tensor([[...]], grad_fn=<AddBackward0>)}

    ```
    """

    def __init__(
        self, module: Module | dict, input_keys: Sequence[str], output_keys: Sequence[str]
    ) -> None:
        super().__init__()
        self.module = setup_module(module)
        self.input_keys = tuple(input_keys)
        self.output_keys = tuple(output_keys)

    def extra_repr(self) -> str:
        return str_mapping({"input_keys": self.input_keys, "output_keys": self.output_keys})

    def forward(self, batch: dict) -> dict:
        out = self.module(*tuple(batch[key] for key in self.input_keys))
        if torch.is_tensor(out):
            out = (out,)
        return {output_key: out[i] for i, output_key in enumerate(self.output_keys)}
