import logging
from pathlib import Path
from typing import Generator, Iterable

logger = logging.getLogger(__name__)

def count_up_to(n: int) -> Generator[int, None, None]:
    """Generates numbers from 0 up to n (exclusive)."""
    for i in range(n):
        logger.debug(f"Yielding {i}")
        yield i
    logger.debug(f"Finished generating numbers up to {n}.")

def read_file_line_by_line(file_path: Path) -> Generator[str, None, None]:
    """Read a file line-by-line without loading entire file into memory."""
    if not file_path.is_file():
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")
    logger.info(f"Opening file {file_path} for reading.")
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            logger.debug(f"Yielding line {line_num}")
            yield line.rstrip("\n")
    logger.info(f"Finished reading {file_path}")


def fibonacci(max_value: int) -> Generator[int, None, None]:
    """Generate Fibonacci sequence up to max_value.

    Args:
        max_value: Upper limit for Fibonacci numbers.

    Yields:
        Fibonacci numbers less than or equal to max_value.
    """
    logger.debug(f"Starting Fibonacci generation, max={max_value}")
    a, b = 0, 1
    count = 0
    while a <= max_value:
        logger.debug(f"Yielding Fibonacci({count})={a}")
        yield a
        a, b = b, a + b
        count += 1
    logger.debug(f"Finished Fibonacci, generated {count} numbers")


def filter_lines(
    lines: Iterable[str], keyword: str
) -> Generator[str, None, None]:
    """Filter lines containing a keyword (generator expression).

    Args:
        lines: Iterable of strings.
        keyword: String to search for.

    Yields:
        Lines containing keyword (case-insensitive).
    """
    logger.debug(f"Filtering for keyword: '{keyword}'")
    count = 0
    for line in lines:
        if keyword.lower() in line.lower():
            logger.debug(f"Matched: {line[:50]}")
            yield line
            count += 1
    logger.info(f"Filter matched {count} lines")


def chunk_list(
    items: Iterable[object], chunk_size: int
) -> Generator[list[object], None, None]:
    """Split iterable into chunks of specified size.

    Args:
        items: Iterable to chunk.
        chunk_size: Size of each chunk.

    Yields:
        Lists of size chunk_size (last chunk may be smaller).

    Raises:
        ValueError: If chunk_size <= 0.
    """
    if chunk_size <= 0:
        logger.error(f"Invalid chunk_size: {chunk_size}")
        raise ValueError("chunk_size must be positive")

    logger.debug(f"Chunking into size {chunk_size}")
    chunk: list[object] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            logger.debug(f"Yielding chunk of size {chunk_size}")
            yield chunk
            chunk = []
    if chunk:
        logger.debug(f"Yielding final chunk of size {len(chunk)}")
        yield chunk