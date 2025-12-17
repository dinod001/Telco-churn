import logging
import pandas as pd
from enum import Enum
from typing import List, Optional
from abc import ABC, abstractmethod
# from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import MinMaxScaler, MinMaxScalerModel, VectorAssembler
from spark_session import get_or_create_spark_session
from pyspark.ml.linalg import Vector

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureScalingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        pass


class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self, scaler_path, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        # self.scaler = MinMaxScaler()
        self.scaler_model = None
        self.fitted = False
        self.scaler_path = scaler_path

    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        if not columns_to_scale:
            logger.warning("No columns provided for scaling. Returning original DataFrame.")
            return df

        logger.info(f"Starting MinMax scaling for columns: {columns_to_scale}")

        ################# Pandas Code ####################
        # df_copy = df.copy()

        # try:
        #     df_copy[columns_to_scale] = self.scaler.fit_transform(df_copy[columns_to_scale])
        #     self.fitted = True
        #     logger.info(f"MinMax scaling applied successfully.")

        #     # Save scaler
        #     joblib.dump(self.scaler, self.scaler_path)
        #     logger.info(f"MinMax scaler saved to: {self.scaler_path}")

        # except Exception as e:
        #     logger.error(f"Error during MinMax scaling: {e}")
        #     raise

        # return df_copy
        
        ################# Pyspark Code ####################
        try:
            # Check if columns exist
            missing_cols = [c for c in columns_to_scale if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

            # 1. Assemble columns into a vector
            assembler = VectorAssembler(
                inputCols=columns_to_scale, 
                outputCol="features_vec_scaling",
                handleInvalid="skip" # or error
            )
            df_assembled = assembler.transform(df)

            # 2. Apply MinMaxScaler
            scaler = MinMaxScaler(
                inputCol="features_vec_scaling", 
                outputCol="scaled_vec_scaling"
            )
            
            self.scaler_model = scaler.fit(df_assembled)
            df_scaled = self.scaler_model.transform(df_assembled)
            self.fitted = True

            # 3. Save scaler model
            self._save_scaler_internal()

            # 4. Unpack vector back to individual columns
            # We need to overwrite the original columns or create new ones. 
            # The Pandas code overwrites.
            
            # UDF to extract value from Vector at index
            # Note: Vectors in Spark can be Dense or Sparse. float(v[i]) works for both.
            def extract_from_vector(vec, i):
                try:
                     return float(vec[i])
                except:
                     return None
            
            vector_extractor_udf = udf(extract_from_vector, DoubleType())

            for i, col_name in enumerate(columns_to_scale):
                df_scaled = df_scaled.withColumn(
                    col_name, 
                    vector_extractor_udf(col("scaled_vec_scaling"), lit(i))
                )

            # Drop temp columns
            df_final = df_scaled.drop("features_vec_scaling", "scaled_vec_scaling")
            
            logger.info(f"MinMax scaling applied and unpacked successfully.")
            return df_final

        except Exception as e:
            logger.error(f"Error during MinMax scaling: {e}")
            raise

    def get_scaler(self):
        if self.fitted:
            logger.info("Returning fitted MinMax scaler model.")
        else:
            logger.warning("Scaler has not been fitted yet.")
        return self.scaler_model

    def _save_scaler_internal(self):
         # Helper to save during fit
        try:
            if self.scaler_model:
                self.scaler_model.write().overwrite().save(self.scaler_path)
                logger.info(f"MinMax scaler (MinMaxScalerModel) saved to: {self.scaler_path}")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
            # Don't raise here to allow pipeline to continue if saving fails but scaling worked? 
            # Original code raised.
            raise

    def save_scaler(self, path=None):
        path = path or self.scaler_path
        try:
             if self.scaler_model:
                self.scaler_model.write().overwrite().save(path)
                logger.info(f"MinMax scaler saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
            raise

    def load_scaler(self, path=None):
        path = path or self.scaler_path
        try:
            self.scaler_model = MinMaxScalerModel.load(path)
            self.fitted = True
            logger.info(f"MinMax scaler loaded from: {path}")
        except Exception as e:
             logger.error(f"Failed to load scaler: {e}")
             raise
