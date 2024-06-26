r"""Contain utility functions for ``torch.nn.Parameter`` objects."""

from __future__ import annotations

__all__ = ["is_parameter", "is_uninitialized_parameter"]

from typing import Any

from torch.nn import Parameter, UninitializedParameter


def is_parameter(value: Any) -> bool:
    r"""Indicate if the input value is a ``torch.nn.Parameter``.

    Args:
        value: The value to check.

    Returns:
        ``True`` if the input value is a ``torch.nn.Parameter``,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from torch.nn import Parameter, UninitializedParameter
    >>> from astrotech.utils.param import is_parameter
    >>> is_parameter(Parameter(torch.ones(2, 3)))
    True
    >>> is_parameter(UninitializedParameter())
    True
    >>> is_parameter(torch.ones(2, 3))
    False

    ```
    """
    return isinstance(value, Parameter)


def is_uninitialized_parameter(value: Any) -> bool:
    r"""Indicate if the input value is a
    ``torch.nn.UninitializedParameter``.

    Args:
        value: Specifies the value to check.

    Returns:
        ``True`` if the input value is a
            ``torch.nn.UninitializedParameter``, otherwise ``False``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from torch.nn import Parameter, UninitializedParameter
    >>> from astrotech.utils.param import is_uninitialized_parameter
    >>> is_uninitialized_parameter(UninitializedParameter())
    True
    >>> is_uninitialized_parameter(Parameter(torch.ones(2, 3)))
    False
    >>> is_uninitialized_parameter(torch.ones(2, 3))
    False

    ```
    """
    return isinstance(value, UninitializedParameter)
