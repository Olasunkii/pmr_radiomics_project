import pandas as pd
from pathlib import Path
from src.logger import setup_logger
from src.config_loader import ConfigLoader


class DataLoader:
    """
    DataLoader handles loading datasets from disk and performing
    initial validation checks with logging support.
    """

    def __init__(self, data_path, logger=None):
        """
        Initialize DataLoader.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Path to dataset file.
        logger : logging.Logger, optional
            Logger instance. If not provided, a default logger is created.
        """
        self.data_path = Path(data_path)
        config = ConfigLoader()
        log_path = config.get_path("output", "logs_path") / "pipeline.log"
        self.logger = logger or setup_logger(log_path)

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset based on file extension.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        self.logger.info(f"Loading data from: {self.data_path}")

        if not self.data_path.exists():
            self.logger.error(f"File not found: {self.data_path}")
            raise FileNotFoundError(f"File not found: {self.data_path}")

        suffix = self.data_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(self.data_path)

        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(self.data_path)

        elif suffix == ".parquet":
            df = pd.read_parquet(self.data_path)

        else:
            self.logger.error(f"Unsupported file format: {suffix}")
            raise ValueError(f"Unsupported file format: {suffix}")

        self.logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate dataset structure.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.
        """
        self.logger.info("Validating dataset...")
        self.logger.info(f"Dataset shape: {df.shape}")
        self.logger.info(f"Columns: {list(df.columns)}")

        if df.empty:
            self.logger.error("Loaded dataframe is empty.")
            raise ValueError("Loaded dataframe is empty.")

        required_columns = ["study_id", "clinical_diagnosis"]

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            self.logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info("Data validation passed.")