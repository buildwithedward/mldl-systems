"""Tests for generator functions."""

import tempfile
from pathlib import Path

import pytest

from src.generators import chunk_list, fibonacci, filter_lines, read_file_line_by_line


class TestBasicGenerators:
    """Test basic generator functions."""

    def test_fibonacci_happy_path(self) -> None:
        """Test Fibonacci generation."""
        result = list(fibonacci(20))
        assert result == [0, 1, 1, 2, 3, 5, 8, 13]

    def test_fibonacci_zero(self) -> None:
        """Test Fibonacci with max_value=0."""
        result = list(fibonacci(0))
        assert result == [0]


class TestFileGenerators:
    """Test generators that work with files."""

    def test_read_file_line_by_line_happy_path(self) -> None:
        """Test reading file line by line."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("line1\nline2\nline3\n")
            temp_path = Path(f.name)

        try:
            result = list(read_file_line_by_line(temp_path))
            assert result == ["line1", "line2", "line3"]
        finally:
            temp_path.unlink()

    def test_read_file_line_by_line_not_found(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            list(read_file_line_by_line(Path("/nonexistent/file.txt")))


class TestFilterGenerators:
    """Test filter generators."""

    def test_filter_lines_happy_path(self) -> None:
        """Test filtering lines by keyword."""
        lines = ["import sys", "import os", "x = 5", "print(x)"]
        result = list(filter_lines(lines, "import"))
        assert result == ["import sys", "import os"]

    def test_filter_lines_no_matches(self) -> None:
        """Test when no lines match."""
        lines = ["a", "b", "c"]
        result = list(filter_lines(lines, "x"))
        assert result == []


class TestChunkList:
    """Test chunking generator."""

    def test_chunk_list_happy_path(self) -> None:
        """Test chunking list."""
        result = list(chunk_list([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunk_list_exact_multiple(self) -> None:
        """Test chunking with exact multiple."""
        result = list(chunk_list([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_chunk_list_invalid_size(self) -> None:
        """Test error on invalid chunk size."""
        with pytest.raises(ValueError):
            list(chunk_list([1, 2, 3], 0))