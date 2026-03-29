"""Main file analyzer combining all Day 5 concepts."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List

from src.comprehensions import dict_from_lists, nested_list_comprehension
from src.dataclasses_demo import FileStats
from src.decorators import log_calls_decorator, timer_decorator
from src.generators import read_file_line_by_line
from src.pathlib_demo import get_file_info, list_python_files
from src.typing_hints import find_in_list

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of analyzing a directory of files.

    Attributes:
        directory: Directory analyzed.
        total_files: Number of files analyzed.
        total_lines: Total lines across all files.
        files_stats: List of FileStats for each file.
    """

    directory: str
    total_files: int
    total_lines: int
    files_stats: List[FileStats]

    def summary_dict(self) -> Dict[str, object]:
        """Convert result to dictionary using dict comprehension.

        Returns:
            Dictionary with analysis summary.
        """
        logger.debug("Converting result to dictionary")
        return {
            "directory": self.directory,
            "file_count": self.total_files,
            "line_count": self.total_lines,
            "avg_lines_per_file": (
                self.total_lines / self.total_files if self.total_files > 0 else 0
            ),
        }


@log_calls_decorator
@timer_decorator
def analyze_python_file(filepath: Path) -> FileStats:
    """Analyze a single Python file and return statistics.

    Args:
        filepath: Path to Python file.

    Returns:
        FileStats object with analysis results.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not filepath.is_file():
        logger.error(f"Not a file: {filepath}")
        raise FileNotFoundError(f"Not a file: {filepath}")

    logger.info(f"Analyzing {filepath.name}")
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    words = content.split()

    # Extract imports using list comprehension
    imports = [
        line.strip().replace("import ", "").replace("from ", "").split()[0]
        for line in lines
        if "import" in line and not line.strip().startswith("#")
    ]

    # Create FileStats using dataclass
    stats = FileStats(
        filepath=str(filepath),
        line_count=len(lines),
        char_count=len(content),
        word_count=len(words),
        imports=list(set(imports)),
    )
    logger.debug(f"Stats: {stats}")
    return stats


@timer_decorator
def analyze_directory(directory: Path) -> AnalysisResult:
    """Analyze all Python files in directory recursively.

    Args:
        directory: Root directory to analyze.

    Returns:
        AnalysisResult with statistics from all files.

    Raises:
        ValueError: If directory does not exist.
    """
    if not directory.is_dir():
        logger.error(f"Not a directory: {directory}")
        raise ValueError(f"Not a directory: {directory}")

    logger.info(f"Starting analysis of {directory}")

    # Use pathlib to find all Python files
    py_files = list_python_files(directory)
    logger.info(f"Found {len(py_files)} Python files to analyze")

    # Analyze each file
    stats_list: List[FileStats] = []
    for filepath in py_files:
        try:
            stats = analyze_python_file(filepath)
            stats_list.append(stats)
        except Exception as e:
            logger.error(f"Failed to analyze {filepath}: {e}")

    # Use list comprehension to calculate totals
    total_lines = sum([stat.line_count for stat in stats_list])
    total_chars = sum([stat.char_count for stat in stats_list])

    logger.info(f"Analysis complete: {len(stats_list)} files, {total_lines} lines")

    # Sort stats by line count (descending) using comprehension
    sorted_stats = sorted(stats_list, key=lambda s: s.line_count, reverse=True)

    return AnalysisResult(
        directory=str(directory),
        total_files=len(stats_list),
        total_lines=total_lines,
        files_stats=sorted_stats,
    )


def generate_report(result: AnalysisResult) -> Generator[str, None, None]:
    """Generate analysis report as lines (using generator).

    Args:
        result: AnalysisResult to report on.

    Yields:
        Lines of the report.
    """
    logger.debug("Generating report")
    yield "=" * 60
    yield "PYTHON FILE ANALYSIS REPORT"
    yield "=" * 60
    yield f"Directory: {result.directory}"
    yield f"Files analyzed: {result.total_files}"
    yield f"Total lines: {result.total_lines}"
    if result.total_files > 0:
        yield f"Average lines per file: {result.total_lines / result.total_files:.1f}"
    yield ""
    yield "DETAILED STATISTICS:"
    yield "-" * 60

    for stat in result.files_stats:
        yield stat.summary()
    yield "=" * 60