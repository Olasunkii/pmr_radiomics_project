import logging
import os

def setup_logger(log_path: str) -> logging.Logger:
    """
    Set up and return a project-wide logger.

    This function creates the logging directory if it does not exist,
    configures logging to write to a file, and returns a logger instance.

    Parameters
    ----------
    log_path : str
        Path to the directory where log files will be stored.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    
    os.makedirs(log_path, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_path, "pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    return logging.getLogger()