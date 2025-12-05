import pandas as pd
from abc import ABC, abstractmethod
import logging

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass


class CustomBinningStrategy(FeatureBinningStrategy):

    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        logger.info(f"Starting binning for column: {column}")

        # Check if column exists
        if column not in df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            raise KeyError(f"Column '{column}' not found.")

        # Create new binned column
        new_column_name = f"{column}_bins"
        df[new_column_name] = df[column].apply(self.tenure_binning)

        logger.info(f"Created binned column: {new_column_name}")

        # Remove original column
        del df[column]
        logger.info(f"Dropped original column: {column}")

        logger.info(f"Binning completed. New DataFrame shape: {df.shape}")

        return df

    @staticmethod
    def tenure_binning(tenure):
        if tenure <= 12:
            return "Newer"
        elif tenure <= 48:
            return "Medium"
        else:
            return "Long-term"
