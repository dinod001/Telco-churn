import pandas as pd
from abc import ABC, abstractmethod
import logging

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_column: str = 'TotalCharges'):
        self.critical_column = critical_column

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting missing value handling using DropMissingValuesStrategy.")

        # Rows before cleaning
        initial_rows = len(df)

        # Step 1: Drop rows with any NaN
        df = df.dropna()
        after_dropna_rows = len(df)
        logger.info(f"Dropped {initial_rows - after_dropna_rows} rows containing NaN values.")

        # Step 2: Remove rows where critical column is empty string or spaces
        df_cleaned = df[~df[self.critical_column].astype(str).str.strip().eq('')]
        final_rows = len(df_cleaned)

        logger.info(
            f"Dropped {after_dropna_rows - final_rows} rows where "
            f"'{self.critical_column}' was blank/empty."
        )

        # Final summary
        logger.info(
            f"Missing value handling complete. Final shape: {df_cleaned.shape}"
        )

        return df_cleaned
