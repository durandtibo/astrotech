from __future__ import annotations

from minevent import PeriodicCondition

from astrotech.utils.event import ConditionalEventHandler, EventHandler


def hello_handler() -> None:
    r"""Implements a simple handler that prints hello."""
    print("Hello!")  # noqa: T201


##################################
#     Tests for EventHandler     #
##################################


def test_gevent_handler_str() -> None:
    assert repr(EventHandler(hello_handler)).startswith("EventHandler(")


def test_gevent_handler_repr() -> None:
    assert str(EventHandler(hello_handler)).startswith("EventHandler(")


#############################################
#     Tests for ConditionalEventHandler     #
#############################################


def test_gconditional_event_handler_str() -> None:
    assert repr(ConditionalEventHandler(hello_handler, PeriodicCondition(2))).startswith(
        "ConditionalEventHandler("
    )


def test_conditional_event_handler_repr() -> None:
    assert str(ConditionalEventHandler(hello_handler, PeriodicCondition(2))).startswith(
        "ConditionalEventHandler("
    )
