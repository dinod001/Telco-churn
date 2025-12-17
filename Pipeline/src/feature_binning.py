import pandas as pd
from abc import ABC, abstractmethod
import logging
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when
from spark_session import get_or_create_spark_session

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureBinningStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self, spark: Optional[SparkSession] = None):
        super().__init__(spark)

    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        logger.info(f"Starting binning for column: {column}")

        ################# Pandas Code ####################
        # # Check if column exists
        # if column not in df.columns:
        #     logger.error(f"Column '{column}' does not exist in the DataFrame.")
        #     raise KeyError(f"Column '{column}' not found.")

        # # Create new binned column
        # new_column_name = f"{column}_bins"
        # df[new_column_name] = df[column].apply(self.tenure_binning)

        # logger.info(f"Created binned column: {new_column_name}")

        # # Remove original column
        # del df[column]
        # logger.info(f"Dropped original column: {column}")

        # logger.info(f"Binning completed. New DataFrame shape: {df.shape}")

        # return df

        ################# Pyspark Code ####################
        if column not in df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            raise KeyError(f"Column '{column}' not found.")

        new_column_name = f"{column}_bins"

        # Apply binning logic using Spark SQL functions
        df_binned = df.withColumn(
            new_column_name,
            when(col(column) <= 12, "Newer")
            .when(col(column) <= 48, "Medium")
            .otherwise("Long-term")
        )

        logger.info(f"Created binned column: {new_column_name}")

        # Drop original column
        df_final = df_binned.drop(column)
        logger.info(f"Dropped original column: {column}")

        logger.info(f"Binning completed. Columns: {len(df_final.columns)}")
        
        return df_final

    @staticmethod
    def tenure_binning(tenure):
        if tenure <= 12:
            return "Newer"
        elif tenure <= 48:
            return "Medium"
        else:
            return "Long-term"
