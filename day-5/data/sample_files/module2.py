"""Sample module 2."""

import json
import logging


logger = logging.getLogger(__name__)


def read_json(filepath: str) -> dict:
    """Read JSON file."""
    with open(filepath) as f:
        return json.load(f)


def main() -> None:
    """Main function."""
    logger.info("Starting...")
