"""Tests for comprehension functions."""

import pytest

from src.comprehensions import (
    dict_comprehension_with_condition,
    dict_from_lists,
    filter_even_numbers,
    nested_list_comprehension,
    set_unique_words,
    square_numbers,
)


class TestListComprehensions:
    """Test list comprehension functions."""

    def test_filter_even_numbers_happy_path(self) -> None:
        """Test filtering even numbers from list."""
        result = filter_even_numbers([1, 2, 3, 4, 5, 6])
        assert result == [2, 4, 6]

    def test_filter_even_numbers_empty(self) -> None:
        """Test filtering empty list."""
        result = filter_even_numbers([])
        assert result == []

    def test_filter_even_numbers_no_matches(self) -> None:
        """Test when no even numbers exist."""
        result = filter_even_numbers([1, 3, 5, 7])
        assert result == []

    def test_square_numbers_happy_path(self) -> None:
        """Test squaring numbers."""
        result = square_numbers([1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

    def test_square_numbers_empty(self) -> None:
        """Test squaring empty list."""
        result = square_numbers([])
        assert result == []


class TestDictComprehensions:
    """Test dictionary comprehension functions."""

    def test_dict_from_lists_happy_path(self) -> None:
        """Test creating dict from parallel lists."""
        result = dict_from_lists(["a", "b", "c"], [1, 2, 3])
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_dict_from_lists_empty(self) -> None:
        """Test with empty lists."""
        result = dict_from_lists([], [])
        assert result == {}

    def test_dict_from_lists_length_mismatch(self) -> None:
        """Test error on length mismatch."""
        with pytest.raises(ValueError):
            dict_from_lists(["a", "b"], [1, 2, 3])

    def test_dict_comprehension_with_condition_happy_path(self) -> None:
        """Test filtering dict by value threshold."""
        data = {"a": 10, "b": 5, "c": 20, "d": 3}
        result = dict_comprehension_with_condition(data, 10)
        assert result == {"a": 10, "c": 20}

    def test_dict_comprehension_with_condition_empty_result(self) -> None:
        """Test when no values meet threshold."""
        data = {"a": 5, "b": 3}
        result = dict_comprehension_with_condition(data, 10)
        assert result == {}


class TestSetComprehensions:
    """Test set comprehension functions."""

    def test_set_unique_words_happy_path(self) -> None:
        """Test extracting unique words."""
        text = "the quick brown fox jumps over the lazy dog"
        result = set_unique_words(text)
        assert "the" in result
        assert "fox" in result
        assert len(result) == 8

    def test_set_unique_words_empty(self) -> None:
        """Test with empty string."""
        result = set_unique_words("")
        assert result == set()


class TestNestedComprehensions:
    """Test nested list comprehension functions."""

    def test_nested_list_comprehension_happy_path(self) -> None:
        """Test flattening 2D matrix."""
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = nested_list_comprehension(matrix)
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_nested_list_comprehension_empty(self) -> None:
        """Test with empty matrix."""
        result = nested_list_comprehension([])
        assert result == []