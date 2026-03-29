"""Tests for pathlib functions."""

import tempfile
from pathlib import Path

import pytest

from src.pathlib_demo import (
    get_file_info,
    iterate_directory,
    read_file_safe,
    relative_path,
    write_file_safe,
)


class TestPathOperations:
    """Test pathlib operations."""

    def test_read_file_safe_happy_path(self) -> None:
        """Test reading file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = Path(f.name)

        try:
            content = read_file_safe(temp_path)
            assert content == "Hello, World!"
        finally:
            temp_path.unlink()

    def test_read_file_safe_not_found(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            read_file_safe(Path("/nonexistent/file.txt"))

    def test_write_file_safe_happy_path(self) -> None:
        """Test writing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "file.txt"
            write_file_safe(filepath, "test content")
            assert filepath.exists()
            assert filepath.read_text() == "test content"

    def test_get_file_info_happy_path(self) -> None:
        """Test getting file info."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            temp_path = Path(f.name)

        try:
            info = get_file_info(temp_path)
            assert info["size_bytes"] == 4
            assert info["is_file"] is True
            assert info["suffix"] == ""
        finally:
            temp_path.unlink()

    def test_iterate_directory_happy_path(self) -> None:
        """Test iterating directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "file1.txt").touch()
            (tmppath / "file2.txt").touch()

            items = list(iterate_directory(tmppath))
            assert len(items) == 2

    def test_relative_path_happy_path(self) -> None:
        """Test relative path computation."""
        abs_path = Path("/home/user/project/src/main.py")
        rel_result = relative_path(abs_path, Path("/home/user/project"))
        assert str(rel_result) == "src/main.py"