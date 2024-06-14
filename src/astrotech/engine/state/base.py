r"""Contain the base class to implement an engine state."""

from __future__ import annotations

__all__ = ["BaseEngineState"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from objectory import AbstractFactory

if TYPE_CHECKING:
    from minrecord import BaseRecord


class BaseEngineState(ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement an engine state.

    A state should implement the following attributes:

    Example usage:

    ```pycon

    >>> from astrotech.engine.state import EngineState
    >>> state = EngineState()
    >>> state
    EngineState(
      (modules): AssetManager(num_assets=0)
      (records): RecordManager()
      (random_seed): 9984043075503325450
      (max_epochs): 1
      (epoch): -1
      (iteration): -1
    )
    >>> state.epoch  # 0-based, the first epoch is 0. -1 means the training has not started
    >>> state.iteration  # 0-based, the first iteration is 0. -1 means the training has not started
    >>> state.max_epochs  # maximum number of epochs to run
    >>> state.random_seed  # random seed

    ```
    """

    @property
    @abstractmethod
    def epoch(self) -> int:
        r"""Get the epoch value.

        The epoch is 0-based, i.e. the first epoch is 0.
        The value ``-1`` is used to indicate the training has not
        started.

        Returns:
            The epoch value.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.epoch
        -1

        ```
        """

    @property
    @abstractmethod
    def iteration(self) -> int:
        r"""Get the iteration value.

        The iteration is 0-based, i.e. the first iteration is 0.
        The value ``-1`` is used to indicate the training has not
        started.

        Returns:
            The iteration value.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.iteration
        -1

        ```
        """

    @property
    @abstractmethod
    def max_epochs(self) -> int:
        r"""Get the maximum number of training epochs.

        Returns:
            The maximum number of training epochs.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.max_epochs
        1

        ```
        """

    @property
    @abstractmethod
    def random_seed(self) -> int:
        r"""Get the random seed.

        Returns:
            The random seed.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState(random_seed=42)
        >>> state.random_seed
        42

        ```
        """

    @abstractmethod
    def add_record(self, record: BaseRecord, key: str | None = None) -> None:
        r"""Add a record to the state.

        Args:
            record: The recordvto add to the state.
            key: The key to use to store the record. If ``None``,
                the name of the record is used.

        Example usage:

        ```pycon

        >>> from minrecord import MinScalarRecord
        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.add_record(MinScalarRecord("loss"))
        >>> state.add_record(MinScalarRecord("loss"), "my key")

        ```
        """

    @abstractmethod
    def add_module(self, name: str, module: Any) -> None:
        r"""Add a module to the engine state.

        Note that the name should be unique. If the name exists, the
        old module will be overwritten by the new module.

        Args:
            name: The name of the module to add tovthe engine state.
            module: The module to add to the enfine state.

        Example usage:

        ```pycon

        >>> from torch import nn
        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.add_module("model", nn.Linear(4, 6))

        ```
        """

    @abstractmethod
    def get_best_values(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        r"""Get the best value of each metric.

        This method ignores the metrics with empty record and the
        non-comparable record.

        Args:
            prefix: The prefix used to create the dict of best values.
                The goal of this prefix is to generate a name which is
                different from the record name to avoid confusion.
                By default, the returned dict uses the same name as the
                record.
            suffix: The suffix used to create the dict of best values.
                The goal of this suffix is to generate a name which is
                different from the record name to avoid confusion.
                By default, the returned dict uses the same name as the
                record.

        Returns:
            The dict with the best value of each metric.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> from minrecord import MaxScalarRecord
        >>> record = MaxScalarRecord("accuracy")
        >>> record.add_value(23.0)
        >>> record.add_value(42.0)
        >>> state.add_record(record)
        >>> state.get_best_values()
        {'accuracy': 42.0}
        >>> state.get_best_values(prefix="best/")
        {'best/accuracy': 42.0}
        >>> state.get_best_values(suffix="/best")
        {'accuracy/best': 42.0}

        ```
        """

    @abstractmethod
    def get_record(self, key: str) -> BaseRecord:
        r"""Get the record associated to a key.

        Args:
            key: The key of the record to retrieve.

        Returns:
            The record if it exists, otherwise it returns an empty
                record. The created empty record is of type ``Record``.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> from minrecord import MinScalarRecord
        >>> state.add_record(MinScalarRecord("loss"))
        >>> state.get_record("loss")
        MinScalarRecord(name=loss, max_size=10, record=())
        >>> state.get_record("new_record")
        GenericRecord(name=new_record, max_size=10, record=())

        ```
        """

    @abstractmethod
    def get_records(self) -> dict[str, BaseRecord]:
        r"""Get all records store in the state.

        Returns:
            The records with their keys.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> from minrecord import MinScalarRecord
        >>> state.add_record(MinScalarRecord("loss"))
        >>> state.get_records()
        {'loss': MinScalarRecord(name=loss, max_size=10, record=())}

        ```
        """

    @abstractmethod
    def get_module(self, name: str) -> Any:
        r"""Get a module.

        Args:
            name: The module name to get.

        Returns:
            The module

        Raises:
            ValueError: if the module does not exist.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> from torch import nn
        >>> state.add_module("model", nn.Linear(4, 6))
        >>> state.get_module("model")
        Linear(in_features=4, out_features=6, bias=True)

        ```
        """

    @abstractmethod
    def has_record(self, key: str) -> bool:
        r"""Indicate if the state has a record for the given key.

        Args:
            key: The key of the record.

        Returns:
            ``True`` if the record exists, ``False`` otherwise.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> from minrecord import MinScalarRecord
        >>> state.add_record(MinScalarRecord("loss"))
        >>> state.has_record("loss")
        True
        >>> state.has_record("missing_record")
        False

        ```
        """

    @abstractmethod
    def has_module(self, name: str) -> bool:
        r"""Indicate if there is module for the given name.

        Args:
            name: The name to check.

        Returns:
            ``True`` if the module exists, otherwise ``False``.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> from torch import nn
        >>> state.add_module("model", nn.Linear(4, 6))
        >>> state.has_module("model")
        True
        >>> state.has_module("missing_module")
        False

        ```
        """

    @abstractmethod
    def increment_epoch(self, increment: int = 1) -> None:
        r"""Increment the epoch value by the given value.

        Args:
            increment: The increment for thevepoch value.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.epoch
        -1
        >>> # Increment the epoch number by 1.
        >>> state.increment_epoch()
        >>> state.epoch
        0
        >>> # Increment the epoch number by 10.
        >>> state.increment_epoch(10)
        >>> state.epoch
        10

        ```
        """

    @abstractmethod
    def increment_iteration(self, increment: int = 1) -> None:
        r"""Increment the iteration value by the given value.

        Args:
             increment: The increment for the iteration value.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.iteration
        -1
        >>> # Increment the iteration number by 1.
        >>> state.increment_iteration()
        >>> state.iteration
        0
        >>> # Increment the iteration number by 10.
        >>> state.increment_iteration(10)
        >>> state.iteration
        10

        ```
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Load the state values from a dict.

        Args:
            state_dict: A dict containing the state values to load.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.load_state_dict({"epoch": 4, "iteration": 42, "records": {}, "modules": {}})
        >>> state.epoch
        4
        >>> state.iteration
        42

        ```
        """

    @abstractmethod
    def remove_module(self, name: str) -> None:
        r"""Remove a module from the state.

        Args:
            name: The name of the module to remove.

        Raises:
            ValueError: if the module name is not found.

        Example usage:

        ```pycon

        >>> from torch import nn
        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.add_module("model", nn.Linear(4, 6))
        >>> state.has_module("model")
        True
        >>> state.remove_module("model")
        >>> state.has_module("model")
        False

        ```
        """

    @abstractmethod
    def state_dict(self) -> dict:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon

        >>> from astrotech.engine.state import EngineState
        >>> state = EngineState()
        >>> state.state_dict()
        {'epoch': -1, 'iteration': -1, 'records': {}, 'modules': {}}

        ```
        """
