import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Preprocessor handles train-test splitting and feature scaling
    for machine learning models.
    """

    def __init__(self, test_size: float, random_state: int, logger):
        """
        Initialize Preprocessor.

        Parameters
        ----------
        test_size : float
            Proportion of dataset to use as test set.
        random_state : int
            Random seed for reproducibility.
        logger : logging.Logger
            Logger instance.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logger
        self.scaler = StandardScaler()

    def split(self, df: pd.DataFrame):
        """
        Perform stratified train-test split.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        self.logger.info("Splitting data into train and test sets...")

        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.logger.info(f"Train shape: {X_train.shape}")
        self.logger.info(f"Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def scale(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Apply standard scaling (fit on train, transform both).

        Returns
        -------
        X_train_scaled, X_test_scaled
        """
        self.logger.info("Scaling features...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled
    
if __name__ == "__main__":
    from pathlib import Path
    from src.config_loader import ConfigLoader
    from preprocessing.data_loader import DataLoader
    from preprocessing.data_cleaner import DataCleaner

    print("Testing Preprocessor...")

    config = ConfigLoader()
    file_path = config.get_path("data", "raw_file")

    loader = DataLoader(file_path)
    df = loader.load_data()

    cleaner = DataCleaner(logger=loader.logger)
    df_clean = cleaner.encode_target(df)
    df_clean = cleaner.drop_columns(df_clean)
    df_clean = cleaner.handle_missing(df_clean)

    preprocessor = Preprocessor(
        test_size=config.get("model", "test_size"),
        random_state=config.get("model", "random_state"),
        logger=loader.logger
    )

    X_train, X_test, y_train, y_test = preprocessor.split(df_clean)
    X_train_scaled, X_test_scaled = preprocessor.scale(X_train, X_test)

    print("Preprocessing successful.")
    print(X_train.shape, X_test.shape)