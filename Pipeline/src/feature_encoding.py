import logging
import os
import pandas as pd
import json
from enum import Enum
from abc import ABC, abstractmethod
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import joblib

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns

    def encode(self, df):
        logger.info(f"Starting nominal encoding for columns: {self.nominal_columns}")
        df_copy = df.copy()

        for column in self.nominal_columns:
            if column not in df_copy.columns:
                logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
                continue

            try:
                logger.info(f"Encoding column: {column}")
                df_dummies = pd.get_dummies(df_copy[column], prefix=column).astype(int)
                df_copy = pd.concat([df_copy, df_dummies], axis=1)
                del df_copy[column]
                logger.info(f"Column '{column}' encoded successfully.")
            except Exception as e:
                logger.error(f"Error encoding column '{column}': {e}")
                raise

        logger.info(f"Nominal encoding completed. DataFrame shape: {df_copy.shape}")
        return df_copy


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_columns):
        self.ordinal_columns = ordinal_columns
        self.encoder = OrdinalEncoder()  # initialize encoder

    def encode(self, df):
        logger.info(f"Starting ordinal encoding for columns: {self.ordinal_columns}")
        df_copy = df.copy()

        try:
            df_copy[self.ordinal_columns] = self.encoder.fit_transform(df_copy[self.ordinal_columns])
            logger.info(f"Ordinal encoding completed. DataFrame shape: {df_copy.shape}")
        except Exception as e:
            logger.error(f"Error during ordinal encoding: {e}")
            raise

        return df_copy

    def save_encoder(self, path):
        try:
            joblib.dump(self.encoder, path)
            logger.info(f"Ordinal encoder saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save ordinal encoder: {e}")
            raise

    def load_encoder(self, path):
        try:
            self.encoder = joblib.load(path)
            logger.info(f"Ordinal encoder loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load ordinal encoder: {e}")
            raise


class LabelEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, target_column):
        self.target_column = target_column
        self.encoder = LabelEncoder()

    def encode(self, df):
        logger.info(f"Starting label encoding for target column: {self.target_column}")
        df_copy = df.copy()

        try:
            df_copy[self.target_column] = self.encoder.fit_transform(df_copy[self.target_column])
            logger.info(f"Label encoding completed for column: {self.target_column}")
        except Exception as e:
            logger.error(f"Error during label encoding for column '{self.target_column}': {e}")
            raise

        return df_copy

    def save_encoder(self, path):
        try:
            joblib.dump(self.encoder, path)
            logger.info(f"Label encoder saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save label encoder: {e}")
            raise

    def load_encoder(self, path):
        try:
            self.encoder = joblib.load(path)
            logger.info(f"Label encoder loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load label encoder: {e}")
            raise
