import logging

# Configure logging (do this once, at startup)
# asctime - the time the log message was created
# levelname - the log level (e.g., INFO, WARNING, ERROR)
# name - the name of the logger (usually __name__)
# message - the actual log message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Get a logger for your module
logger = logging.getLogger(__name__)

# Use it
logger.info("Application started")
logger.warning("This is deprecated")
logger.error("Failed to load config")