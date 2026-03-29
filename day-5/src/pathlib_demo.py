"""Pathlib: modern file path operations without os.path."""

import logging
from pathlib import Path
from typing import Generator, List

logger = logging.getLogger(__name__)


def list_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory recursively.

    Args:
        directory: Root directory to search.

    Returns:
        List of Path objects for all .py files.

    Raises:
        ValueError: If directory does not exist.
    """
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        raise ValueError(f"Directory does not exist: {directory}")

    logger.info(f"Searching for Python files in {directory}")
    py_files = sorted(directory.glob("**/*.py"))
    logger.info(f"Found {len(py_files)} Python files")
    return py_files


def read_file_safe(filepath: Path) -> str:
    """Read file contents with error handling.

    Args:
        filepath: Path to file.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not filepath.is_file():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Not a file: {filepath}")

    logger.debug(f"Reading file: {filepath}")
    content = filepath.read_text(encoding="utf-8")
    logger.info(f"Read {len(content)} chars from {filepath.name}")
    return content


def write_file_safe(filepath: Path, content: str) -> None:
    """Write content to file, creating parent directories if needed.

    Args:
        filepath: Path to file.
        content: Content to write.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Writing {len(content)} chars to {filepath}")
    filepath.write_text(content, encoding="utf-8")
    logger.info(f"Wrote file: {filepath}")


def get_file_info(filepath: Path) -> dict[str, object]:
    """Get file information (size, type, permissions).

    Args:
        filepath: Path to file.

    Returns:
        Dictionary with file information.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File does not exist: {filepath}")

    logger.debug(f"Getting info for {filepath}")
    stat = filepath.stat()
    info = {
        "name": filepath.name,
        "stem": filepath.stem,
        "suffix": filepath.suffix,
        "parent": str(filepath.parent),
        "size_bytes": stat.st_size,
        "is_file": filepath.is_file(),
        "is_dir": filepath.is_dir(),
        "is_symlink": filepath.is_symlink(),
    }
    logger.debug(f"File info: {info}")
    return info


def iterate_directory(directory: Path) -> Generator[Path, None, None]:
    """Iterate over immediate children of directory.

    Args:
        directory: Directory to iterate.

    Yields:
        Path objects for each item in directory.

    Raises:
        ValueError: If directory does not exist or is not a directory.
    """
    if not directory.is_dir():
        logger.error(f"Not a directory: {directory}")
        raise ValueError(f"Not a directory: {directory}")

    logger.debug(f"Iterating {directory}")
    for item in directory.iterdir():
        logger.debug(f"Found item: {item.name}")
        yield item


def relative_path(filepath: Path, relative_to: Path) -> Path:
    """Convert absolute path to relative path.

    Args:
        filepath: Absolute path.
        relative_to: Reference directory.

    Returns:
        Relative path from relative_to to filepath.
    """
    logger.debug(f"Computing relative path: {filepath} relative to {relative_to}")
    result = filepath.relative_to(relative_to)
    logger.debug(f"Relative path: {result}")
    return result