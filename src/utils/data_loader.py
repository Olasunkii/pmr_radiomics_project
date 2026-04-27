import pandas as pd
import os


class DataLoader:
    """
    DataLoader is responsible for loading raw datasets and performing
    initial validation checks.

    This class ensures that the input data is correctly structured
    before further preprocessing steps.
    """

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Parameters:
        ----------
        data_path : str
            Path to the CSV file.
        """
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV file.

        Returns:
        -------
        pd.DataFrame
            Loaded dataset.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        return df

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Perform basic validation checks on the dataset.

        Checks:
        - Empty dataframe
        - Presence of key columns

        Parameters:
        ----------
        df : pd.DataFrame
            Dataset to validate.
        """
        if df.empty:
            raise ValueError("Loaded dataframe is empty.")

        required_columns = ["study_id", "clinical_diagnosis"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        print("Data validation passed.")