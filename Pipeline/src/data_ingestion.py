import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from pyspark.sql import SparkSession
from spark_session import get_or_create_spark_session
import logging

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---- Abstract Base Class ----
class DataIngestion(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()
    @abstractmethod
    def ingest(self):
        pass


# ---- CSV Reader ----
class ReadCSV(DataIngestion):
    def __init__(self, data_path: str, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.data_path = data_path

    def ingest(self):
        logger.info(f"Reading CSV file from: {self.data_path}")
        try:
            ################# Pandas Code ####################
            # df = pd.read_csv(self.data_path)
            # logger.info(f"CSV successfully read. Shape: {df.shape}")
            # return df

            ################# Pyspark Code ####################
            df = self.spark.read.csv(self.data_path, header=True, inferSchema=True)
            logger.info(f"CSV successfully read. Rows: {df.count()}, Columns: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise


# ---- Excel Reader ----
class ReadExcel(DataIngestion):
    def __init__(self, data_path: str, sheet_name=0, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.data_path = data_path
        self.sheet_name = sheet_name

    def ingest(self):
        logger.info(f"Reading Excel file from: {self.data_path}, sheet: {self.sheet_name}")
        try:
            ################# Pandas Code ####################
            # df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
            # logger.info(f"Excel successfully read. Shape: {df.shape}")
            # return df

            ################# Pyspark Code ####################
            pandas_df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
            df = self.spark.createDataFrame(pandas_df)
            logger.info(f"Excel successfully read. Rows: {df.count()}, Columns: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise
