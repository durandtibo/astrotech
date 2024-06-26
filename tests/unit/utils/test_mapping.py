from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from astrotech.utils.mapping import (
    convert_to_dict_of_lists,
    convert_to_list_of_dicts,
    get_first_value,
    remove_keys_starting_with,
    to_flat_dict,
)

if is_numpy_available():
    import numpy as np


##############################################
#     Tests for convert_to_dict_of_lists     #
##############################################


def test_convert_to_dict_of_lists_empty_list() -> None:
    assert convert_to_dict_of_lists([]) == {}


def test_convert_to_dict_of_lists_empty_dict() -> None:
    assert convert_to_dict_of_lists([{}]) == {}


def test_convert_to_dict_of_lists() -> None:
    assert objects_are_equal(
        convert_to_dict_of_lists(
            [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
        ),
        {
            "key1": [1, 2, 3],
            "key2": [10, 20, 30],
        },
    )


##############################################
#     Tests for convert_to_list_of_dicts     #
##############################################


def test_convert_to_list_of_dicts_empty_dict() -> None:
    assert convert_to_list_of_dicts({}) == []


def test_convert_to_list_of_dicts_empty_list() -> None:
    assert convert_to_list_of_dicts({"key1": [], "key2": []}) == []


def test_convert_to_list_of_dicts() -> None:
    assert objects_are_equal(
        convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]}),
        [
            {"key1": 1, "key2": 10},
            {"key1": 2, "key2": 20},
            {"key1": 3, "key2": 30},
        ],
    )


###############################################
#     Tests for remove_keys_starting_with     #
###############################################


def test_remove_keys_starting_with_empty() -> None:
    assert remove_keys_starting_with({}, "key") == {}


def test_remove_keys_starting_with() -> None:
    assert objects_are_equal(
        remove_keys_starting_with(
            {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key"
        ),
        {
            "abc": 3,
            "abc.key": 4,
            1: 5,
            (2, 3): 6,
        },
    )


def test_remove_keys_starting_with_another_key() -> None:
    assert objects_are_equal(
        remove_keys_starting_with(
            {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key."
        ),
        {
            "key": 1,
            "abc": 3,
            "abc.key": 4,
            1: 5,
            (2, 3): 6,
        },
    )


#####################################
#     Tests for get_first_value     #
#####################################


def test_get_first_value_empty() -> None:
    with pytest.raises(
        ValueError, match="First value cannot be returned because the mapping is empty"
    ):
        get_first_value({})


def test_get_first_value() -> None:
    assert get_first_value({"key1": 1, "key2": 2}) == 1


##################################
#     Tests for to_flat_dict     #
##################################


def test_to_flat_dict_flat_dict() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "bool": False,
                "float": 3.5,
                "int": 2,
                "str": "abc",
            }
        ),
        {
            "bool": False,
            "float": 3.5,
            "int": 2,
            "str": "abc",
        },
    )


def test_to_flat_dict_nested_dict_str() -> None:
    assert objects_are_equal(
        to_flat_dict({"a": "a", "b": {"c": "c"}, "d": {"e": {"f": "f"}}}),
        {"a": "a", "b.c": "c", "d.e.f": "f"},
    )


def test_to_flat_dict_nested_dict_multiple_types() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": {
                    "bool": False,
                    "float": 3.5,
                    "int": 2,
                },
                "str": "abc",
            }
        ),
        {
            "module.bool": False,
            "module.float": 3.5,
            "module.int": 2,
            "str": "abc",
        },
    )


def test_to_flat_dict_data_empty_key() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": {},
                "str": "abc",
            }
        ),
        {"str": "abc"},
    )


def test_to_flat_dict_double_data() -> None:
    assert objects_are_equal(
        {
            "module.component.float": 3.5,
            "module.component.int": 2,
            "str": "def",
        },
        to_flat_dict(
            {
                "str": "def",
                "module": {
                    "component": {
                        "float": 3.5,
                        "int": 2,
                    },
                },
            }
        ),
    )


def test_to_flat_dict_double_data_2() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": {
                    "component_a": {
                        "float": 3.5,
                        "int": 2,
                    },
                    "component_b": {
                        "param_a": 1,
                        "param_b": 2,
                    },
                    "str": "abc",
                },
            }
        ),
        {
            "module.component_a.float": 3.5,
            "module.component_a.int": 2,
            "module.component_b.param_a": 1,
            "module.component_b.param_b": 2,
            "module.str": "abc",
        },
    )


def test_to_flat_dict_list() -> None:
    assert objects_are_equal(
        to_flat_dict([2, "abc", True, 3.5]),
        {
            "0": 2,
            "1": "abc",
            "2": True,
            "3": 3.5,
        },
    )


def test_to_flat_dict_dict_with_list() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": [2, "abc", True, 3.5],
                "str": "abc",
            }
        ),
        {
            "module.0": 2,
            "module.1": "abc",
            "module.2": True,
            "module.3": 3.5,
            "str": "abc",
        },
    )


def test_to_flat_dict_with_more_complex_list() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": [[1, 2, 3], {"bool": True}],
                "str": "abc",
            }
        ),
        {
            "module.0.0": 1,
            "module.0.1": 2,
            "module.0.2": 3,
            "module.1.bool": True,
            "str": "abc",
        },
    )


def test_to_flat_dict_tuple() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": (2, "abc", True, 3.5),
                "str": "abc",
            }
        ),
        {
            "module.0": 2,
            "module.1": "abc",
            "module.2": True,
            "module.3": 3.5,
            "str": "abc",
        },
    )


def test_to_flat_dict_with_complex_tuple() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": ([1, 2, 3], {"bool": True}),
                "str": "abc",
            }
        ),
        {
            "module.0.0": 1,
            "module.0.1": 2,
            "module.0.2": 3,
            "module.1.bool": True,
            "str": "abc",
        },
    )


@pytest.mark.parametrize("separator", [".", "/", "@", "[SEP]"])
def test_to_flat_dict_separator(separator: str) -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "str": "def",
                "module": {
                    "component": {
                        "float": 3.5,
                        "int": 2,
                    },
                },
            },
            separator=separator,
        ),
        {
            f"module{separator}component{separator}float": 3.5,
            f"module{separator}component{separator}int": 2,
            "str": "def",
        },
    )


def test_to_flat_dict_to_str_tuple() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module": (2, "abc", True, 3.5),
                "str": "abc",
            },
            to_str=tuple,
        ),
        {
            "module": "(2, 'abc', True, 3.5)",
            "str": "abc",
        },
    )


def test_to_flat_dict_to_str_tuple_and_list() -> None:
    assert objects_are_equal(
        to_flat_dict(
            {
                "module1": (2, "abc", True, 3.5),
                "module2": [1, 2, 3],
                "str": "abc",
            },
            to_str=(list, tuple),
        ),
        {
            "module1": "(2, 'abc', True, 3.5)",
            "module2": "[1, 2, 3]",
            "str": "abc",
        },
    )


def test_to_flat_dict_tensor() -> None:
    assert objects_are_equal(
        to_flat_dict({"tensor": torch.ones(2, 3)}), {"tensor": torch.ones(2, 3)}
    )


@numpy_available
def test_to_flat_dict_numpy_ndarray() -> None:
    assert objects_are_equal(to_flat_dict(np.zeros((2, 3))), {None: np.zeros((2, 3))})
