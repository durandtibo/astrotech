r"""Contain a simple asset manager implementation."""

from __future__ import annotations

__all__ = ["AssetExistsError", "AssetManager", "AssetNotFoundError"]

import copy
import logging
from typing import Any

from coola import objects_are_equal, summary
from coola.utils import str_indent, str_mapping

logger = logging.getLogger(__name__)


class AssetExistsError(Exception):
    r"""Raised when trying to add an asset that already exists."""


class AssetNotFoundError(Exception):
    r"""Raised when trying to access an asset that does not exist."""


class AssetManager:
    r"""Implement a simple asset manager.

    Example usage:

    ```pycon

    >>> from astrotech.utils.asset import AssetManager
    >>> manager = AssetManager()
    >>> manager
    AssetManager(num_assets=0)
    >>> manager.add_asset("mean", 5)
    >>> manager
    AssetManager(num_assets=1)
    >>> manager.get_asset("mean")
    5

    ```
    """

    def __init__(self, assets: dict[str, Any] | None = None) -> None:
        self._assets = assets or {}

    def __len__(self) -> int:
        return len(self._assets)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_assets={len(self._assets):,})"

    def __str__(self) -> str:
        assets = {name: summary(asset) for name, asset in self._assets.items()}
        args = f"\n  {str_indent(str_mapping(assets))}\n" if assets else ""
        return f"{self.__class__.__qualname__}({args})"

    def add_asset(self, name: str, asset: Any, replace_ok: bool = False) -> None:
        r"""Add an asset to the asset manager.

        Note that the name should be unique. If the name exists, the
        old asset will be overwritten by the new asset.

        Args:
            name: The name of the asset to add.
            asset: The asset to add.
            replace_ok: If ``False``, ``AssetExistsError`` is raised
                if an asset with the same name exists.

        Example usage:

        ```pycon

        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager.add_asset("mean", 5)

        ```
        """
        if name in self._assets and not replace_ok:
            msg = (
                f"`{name}` is already used to register an asset. "
                "Use `replace_ok=True` to replace an asset"
            )
            raise AssetExistsError(msg)
        self._assets[name] = asset

    def clone(self) -> AssetManager:
        r"""Return a deep copy of the current asset manager.

        Returns:
            A deep copy of the current asset manager.

        Example usage:

        ```pycon

        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager({"name": 5})
        >>> cloned = manager.clone()
        >>> manager.add_asset("name", 7, replace_ok=True)
        >>> print(manager)
        AssetManager(
          (name): <class 'int'>  7
        )
        >>> print(cloned)
        AssetManager(
          (name): <class 'int'>  5
        )

        ```
        """
        return AssetManager(copy.deepcopy(self._assets))

    def equal(self, other: Any) -> bool:
        r"""Indicate if two objects are equal.

        Args:
            other: The object to compare with.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager.add_asset("mean", 5)
        >>> manager2 = AssetManager()
        >>> manager.equal(manager2)
        False
        >>> manager2.add_asset("mean", 5)
        >>> manager.equal(manager2)
        True

        ```
        """
        if not isinstance(other, AssetManager):
            return False
        return objects_are_equal(self._assets, other._assets)

    def get_asset(self, name: str) -> Any:
        r"""Get an asset.

        Args:
            name: The name of the asset to get.

        Returns:
            The asset associated to the name.

        Raises:
            AssetNotFoundError: if the asset does not exist.

        Example usage:

        pycon

            >>> from astrotech.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("mean", 5)
            >>> manager.get_asset("mean")
            5
        """
        if name not in self._assets:
            msg = f"The asset '{name}' does not exist"
            raise AssetNotFoundError(msg)
        return self._assets[name]

    def get_asset_names(self) -> tuple[str, ...]:
        r"""Get all the asset names.

        Returns:
            The asset names.

        Example usage:

        ```pycon

        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager.add_asset("mean", 5)
        >>> manager.get_asset_names()
        ('mean',)

        ```
        """
        return tuple(self._assets.keys())

    def has_asset(self, name: str) -> bool:
        r"""Indicate if the asset exists or not.

        Args:
            name: The name of the asset.

        Returns:
            ``True`` if the asset exists, otherwise ``False``

        Example usage:

        ```pycon

        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager.has_asset("mean")
        False
        >>> manager.add_asset("mean", 5)
        >>> manager.has_asset("mean")
        True

        ```
        """
        return name in self._assets

    def remove_asset(self, name: str) -> None:
        r"""Remove an asset.

        Args:
            name: The name of the asset to remove.

        Raises:
            AssetNotFoundError: if the asset does not exist.

        Example usage:

        ```pycon

        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager.add_asset("mean", 5)
        >>> manager.remove_asset("mean")
        >>> manager.has_asset("mean")
        False

        ```
        """
        if name not in self._assets:
            msg = f"The asset '{name}' does not exist so it is not possible to remove it"
            raise AssetNotFoundError(msg)
        del self._assets[name]

    def load_state_dict(self, state_dict: dict, keys: list | tuple | None = None) -> None:
        r"""Load the state dict of each module.

        Note this method ignore the missing modules or keys. For
        example if you want to load the optimizer module but there is
        no 'optimizer' key in the state dict, this method will ignore
        the optimizer module.

        Args:
            state_dict: The state dict to load.
            keys: The keys to load. If ``None``, it loads all the keys
                associated to the registered modules.

        Example usage:

        ```pycon

        >>> import torch
        >>> from torch import nn
        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager.add_asset("my_module", nn.Linear(4, 6))
        >>> manager.load_state_dict(
        ...     {"my_module": {"weight": torch.ones(6, 4), "bias": torch.zeros(6)}}
        ... )

        ```
        """
        keys = keys or tuple(self._assets.keys())
        for key in keys:
            if key not in state_dict:
                logger.info(f"Ignore key {key} because it is not in the state dict")
                continue
            if key not in self._assets:
                logger.info(f"Ignore key {key} because there is no module associated to it")
                continue
            if not hasattr(self._assets[key], "load_state_dict"):
                logger.info(
                    f"Ignore key {key} because the module does not have 'load_state_dict' method"
                )
                continue
            self._assets[key].load_state_dict(state_dict[key])

    def state_dict(self) -> dict:
        r"""Return a state dict with all the modules.

        The state of each module is store with the associated key of
        the module.

        Returns:
            The state dict of all the modules.

        Example usage:

        ```pycon

        >>> from torch import nn
        >>> from astrotech.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager.add_asset("my_module", nn.Linear(4, 6))
        >>> manager.state_dict()
        {'my_module': ...}
        >>> manager.add_asset("int", 123)
        >>> manager.state_dict()
        {'my_module': ...}

        ```
        """
        state = {}
        for name, module in self._assets.items():
            if hasattr(module, "state_dict"):
                state[name] = module.state_dict()
            else:
                logger.info(f"Skip '{name}' module because it does not have 'state_dict' method")
        return state
