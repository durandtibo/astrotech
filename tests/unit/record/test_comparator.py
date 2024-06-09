from __future__ import annotations

from astrotech.record import MaxScalarComparator, MinScalarComparator

#########################################
#     Tests for MaxScalarComparator     #
#########################################


def test_max_scalar_equal_true() -> None:
    assert MaxScalarComparator().equal(MaxScalarComparator())


def test_max_scalar_equal_false() -> None:
    assert not MaxScalarComparator().equal(MinScalarComparator())


def test_max_scalar_get_initial_best_value() -> None:
    assert MaxScalarComparator().get_initial_best_value() == -float("inf")


def test_max_scalar_is_better_int() -> None:
    comparator = MaxScalarComparator()
    assert comparator.is_better(5, 12)
    assert comparator.is_better(12, 12)
    assert not comparator.is_better(12, 5)


def test_max_scalar_is_better_float() -> None:
    comparator = MaxScalarComparator()
    assert comparator.is_better(5.2, 12.1)
    assert comparator.is_better(5.2, 5.2)
    assert not comparator.is_better(12.2, 5.1)


#########################################
#     Tests for MinScalarComparator     #
#########################################


def test_min_scalar_equal_true() -> None:
    assert MinScalarComparator().equal(MinScalarComparator())


def test_min_scalar_equal_false() -> None:
    assert not MinScalarComparator().equal(MaxScalarComparator())


def test_min_scalar_get_initial_best_value() -> None:
    assert MinScalarComparator().get_initial_best_value() == float("inf")


def test_min_scalar_is_better_int() -> None:
    comparator = MinScalarComparator()
    assert not comparator.is_better(5, 12)
    assert comparator.is_better(12, 12)
    assert comparator.is_better(12, 5)


def test_min_scalar_is_better_float() -> None:
    comparator = MinScalarComparator()
    assert not comparator.is_better(5.2, 12.1)
    assert comparator.is_better(5.2, 5.2)
    assert comparator.is_better(12.2, 5.1)
