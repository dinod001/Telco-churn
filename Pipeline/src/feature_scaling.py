import logging
import pandas as pd
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureScalingStrategy(ABC):

    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass


class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self,scaler_path):
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.scaler_path = scaler_path

    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        if not columns_to_scale:
            logger.warning("No columns provided for scaling. Returning original DataFrame.")
            return df

        logger.info(f"Starting MinMax scaling for columns: {columns_to_scale}")

        df_copy = df.copy()

        try:
            df_copy[columns_to_scale] = self.scaler.fit_transform(df_copy[columns_to_scale])
            self.fitted = True
            logger.info(f"MinMax scaling applied successfully.")

            # Save scaler
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"MinMax scaler saved to: {self.scaler_path}")

        except Exception as e:
            logger.error(f"Error during MinMax scaling: {e}")
            raise

        return df_copy

    def get_scaler(self):
        if self.fitted:
            logger.info("Returning fitted MinMax scaler.")
        else:
            logger.warning("Scaler has not been fitted yet. Returning current scaler object.")
        return self.scaler
