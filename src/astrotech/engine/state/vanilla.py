r"""Contain the default engine state implementation."""

from __future__ import annotations

__all__ = ["EngineState"]

from typing import Any

from coola.utils import str_indent, str_mapping
from minrecord import BaseRecord, RecordManager

from astrotech.engine.state.base import BaseEngineState
from astrotech.utils.asset import AssetManager


class EngineState(BaseEngineState):
    r"""Define the default engine state.

    Args:
        epoch: The number of epochsvperformed.
        iteration: The number of training iterations performed.
        max_epochs: The maximum number of epochs.
        random_seed: The random seed.

    Example usage:

    ```pycon

    >>> from astrotech.engine.state import EngineState
    >>> state = EngineState()
    >>> state
    EngineState(
      (epoch): -1
      (iteration): -1
      (max_epochs): 1
      (modules): AssetManager(num_assets=0)
      (random_seed): 9984043075503325450
      (records): RecordManager()
    )

    ```
    """

    def __init__(
        self,
        epoch: int = -1,
        iteration: int = -1,
        max_epochs: int = 1,
        random_seed: int = 9984043075503325450,
    ) -> None:
        self._epoch = epoch
        self._iteration = iteration
        self._max_epochs = max_epochs
        self._random_seed = random_seed

        self._records = RecordManager()
        self._modules = AssetManager()

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "epoch": self._epoch,
                    "iteration": self._iteration,
                    "max_epochs": self._max_epochs,
                    "modules": self._modules,
                    "random_seed": self._random_seed,
                    "records": self._records,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def random_seed(self) -> int:
        return self._random_seed

    def add_record(self, record: BaseRecord, key: str | None = None) -> None:
        self._records.add_record(record=record, key=key)

    def add_module(self, name: str, module: Any) -> None:
        self._modules.add_asset(name=name, asset=module, replace_ok=True)

    def get_best_values(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        return self._records.get_best_values(prefix, suffix)

    def get_record(self, key: str) -> BaseRecord:
        return self._records.get_record(key)

    def get_records(self) -> dict[str, BaseRecord]:
        return self._records.get_records()

    def get_module(self, name: str) -> Any:
        return self._modules.get_asset(name)

    def has_record(self, key: str) -> bool:
        return self._records.has_record(key)

    def has_module(self, name: str) -> bool:
        return self._modules.has_asset(name)

    def increment_epoch(self, increment: int = 1) -> None:
        self._epoch += increment

    def increment_iteration(self, increment: int = 1) -> None:
        self._iteration += increment

    def load_state_dict(self, state_dict: dict) -> None:
        self._epoch = state_dict["epoch"]
        self._iteration = state_dict["iteration"]
        self._records.load_state_dict(state_dict["records"])
        self._modules.load_state_dict(state_dict["modules"])

    def remove_module(self, name: str) -> None:
        self._modules.remove_asset(name)

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "iteration": self._iteration,
            "records": self._records.state_dict(),
            "modules": self._modules.state_dict(),
        }
