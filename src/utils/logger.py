import logging
from pathlib import Path


def setup_logger(log_file: Path = None) -> logging.Logger:
    """
    Configure and return a project-wide logger.

    Parameters
    ----------
    log_file : pathlib.Path, optional
        Path to a log file. If provided, logs will also be written to file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("pmr_project")

    if logger.hasHandlers():
        return logger  # Avoid duplicate handlers

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
