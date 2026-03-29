"""Tests for the main file analyzer."""

import tempfile
from pathlib import Path

import pytest

from src.file_analyzer import analyze_directory, analyze_python_file


class TestFileAnalyzer:
    """Test file analyzer functions."""

    def test_analyze_python_file_happy_path(self) -> None:
        """Test analyzing a Python file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import os\nimport sys\n\nx = 42\nprint(x)\n")
            temp_path = Path(f.name)

        try:
            stats = analyze_python_file(temp_path)
            assert stats.line_count == 5
            assert stats.char_count > 0
            assert len(stats.imports) > 0
        finally:
            temp_path.unlink()

    def test_analyze_python_file_not_found(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            analyze_python_file(Path("/nonexistent/file.py"))

    def test_analyze_directory_happy_path(self) -> None:
        """Test analyzing a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "file1.py").write_text("import os\nx = 1\n")
            (tmppath / "file2.py").write_text("import sys\ny = 2\n")

            result = analyze_directory(tmppath)
            assert result.total_files == 2
            assert result.total_lines > 0

    def test_analyze_directory_not_found(self) -> None:
        """Test error on missing directory."""
        with pytest.raises(ValueError):
            analyze_directory(Path("/nonexistent/dir"))