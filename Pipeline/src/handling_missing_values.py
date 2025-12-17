import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from spark_session import get_or_create_spark_session
from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.functions import col, trim
import logging

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MissingValueHandlingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()
    @abstractmethod
    def handle(self, df: DataFrame) -> DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_column: str = 'TotalCharges', spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.critical_column = critical_column

    def handle(self, df: DataFrame) -> DataFrame:
        logger.info("Starting missing value handling using DropMissingValuesStrategy.")

        ################## Pandas Code ##################

        # # Rows before cleaning
        # initial_rows = len(df)

        # # Step 1: Drop rows with any NaN
        # df = df.dropna()
        # after_dropna_rows = len(df)
        # logger.info(f"Dropped {initial_rows - after_dropna_rows} rows containing NaN values.")

        # # Step 2: Remove rows where critical column is empty string or spaces
        # df_cleaned = df[~df[self.critical_column].astype(str).str.strip().eq('')]
        # final_rows = len(df_cleaned)

        # logger.info(
        #     f"Dropped {after_dropna_rows - final_rows} rows where "
        #     f"'{self.critical_column}' was blank/empty."
        # )

        # # Final summary
        # logger.info(
        #     f"Missing value handling complete. Final shape: {df_cleaned.shape}"
        # )

        ################## Spark Code ##################
        # Rows before cleaning
        initial_rows = df.count()

        # Step 1: Drop rows with any NaN
        df = df.dropna()
        after_dropna_rows = df.count()
        logger.info(f"Dropped {initial_rows - after_dropna_rows} rows containing NaN values.")

        # Step 2: Remove rows where critical column is empty string or spaces
        df_cleaned = df.filter(
            col(self.critical_column).isNotNull() &
            (trim(col(self.critical_column)) != "")
        )
        final_rows = df_cleaned.count()

        logger.info(
            f"Dropped {after_dropna_rows - final_rows} rows where "
            f"'{self.critical_column}' was blank/empty."
        )

        # Final summary
        logger.info(
            f"Missing value handling complete. Rows: {final_rows}, Columns: {len(df_cleaned.columns)}"
        )

        return df_cleaned
