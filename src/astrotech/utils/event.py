r"""Contain utility functions to manage events."""

from __future__ import annotations

__all__ = ["ConditionalEventHandler", "EventHandler"]

from coola.utils import str_indent, str_mapping
from minevent import ConditionalEventHandler as ConditionalEventHandler_
from minevent import EventHandler as EventHandler_


class EventHandler(EventHandler_):
    r"""Implements a variant of ``minvent.EventHandler`` to not show the
    arguments in the to string method.

    Example usage:

    ```pycon

    >>> from astrotech.utils.event import EventHandler
    >>> def hello_handler() -> None:
    ...     print("Hello!")
    ...
    >>> handler = EventHandler(hello_handler)
    >>> print(repr(handler))
    EventHandler(
      (handler): <function hello_handler at 0x...>
      (handler_args): ()
      (handler_kwargs): {}
    )
    >>> print(str(handler))
    EventHandler(
      (handler): <function hello_handler at 0x...>
    )
    >>> handler.handle()
    Hello!

    ```
    """

    def __str__(self) -> str:
        args = str_indent(str_mapping({"handler": self._handler}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class ConditionalEventHandler(ConditionalEventHandler_):
    r"""Implements a variant of ``minvent.ConditionalEventHandler`` to
    not show the arguments in the to string method.

    Example usage:

    ```pycon

    >>> from minevent import PeriodicCondition
    >>> from astrotech.utils.event import ConditionalEventHandler
    >>> def hello_handler() -> None:
    ...     print("Hello!")
    ...
    >>> handler = ConditionalEventHandler(hello_handler, PeriodicCondition(freq=3))
    >>> print(repr(handler))
    ConditionalEventHandler(
      (handler): <function hello_handler at 0x...>
      (handler_args): ()
      (handler_kwargs): {}
      (condition): PeriodicCondition(freq=3, step=0)
    )
    >>> print(str(handler))
    ConditionalEventHandler(
      (handler): <function hello_handler at 0x...>
      (condition): PeriodicCondition(freq=3, step=0)
    )
    >>> handler.handle()
    Hello!
    >>> handler.handle()
    >>> handler.handle()
    >>> handler.handle()
    Hello!

    ```
    """

    def __str__(self) -> str:
        args = str_indent(str_mapping({"handler": self._handler, "condition": self._condition}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
