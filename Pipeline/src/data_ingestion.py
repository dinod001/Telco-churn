import pandas as pd
from abc import ABC, abstractmethod
import logging

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---- Abstract Base Class ----
class DataIngestion(ABC):
    @abstractmethod
    def ingest(self):
        pass


# ---- CSV Reader ----
class ReadCSV(DataIngestion):
    def __init__(self, data_path: str):
        self.data_path = data_path

    def ingest(self):
        logger.info(f"Reading CSV file from: {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"CSV successfully read. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise


# ---- Excel Reader ----
class ReadExcel(DataIngestion):
    def __init__(self, data_path: str, sheet_name=0):
        self.data_path = data_path
        self.sheet_name = sheet_name

    def ingest(self):
        logger.info(f"Reading Excel file from: {self.data_path}, sheet: {self.sheet_name}")
        try:
            df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
            logger.info(f"Excel successfully read. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise
