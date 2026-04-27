import pandas as pd


class DataCleaner:
    """
    DataCleaner handles preprocessing of raw data into a format
    suitable for machine learning models.
    """

    def __init__(self, logger):
        """
        Initialize DataCleaner.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for tracking operations.
        """
        self.logger = logger

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode clinical diagnosis into binary target.

        Returns
        -------
        pd.DataFrame
        """
        self.logger.info("Encoding target variable...")

        df["target"] = df["clinical_diagnosis"].map({
            "PMR": 1,
            "Non-PMR": 0
        })

        if df["target"].isnull().any():
            raise ValueError("Target encoding failed.")

        return df

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop non-informative columns.
        """
        self.logger.info("Dropping identifier columns...")

        df = df.drop(columns=["study_id", "clinical_diagnosis"], errors="ignore")

        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values.
        """
        self.logger.info("Handling missing values...")

        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.mean(numeric_only=True))
            self.logger.info("Missing values filled with mean.")
        else:
            self.logger.info("No missing values found.")

        return df