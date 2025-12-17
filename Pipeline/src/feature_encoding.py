import logging
import os
import pandas as pd
import json
from enum import Enum
from abc import ABC, abstractmethod
# from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import joblib
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from spark_session import get_or_create_spark_session

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def encode(self, df: DataFrame) -> DataFrame:
        pass


class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.nominal_columns = nominal_columns

    def encode(self, df: DataFrame) -> DataFrame:
        logger.info(f"Starting nominal encoding for columns: {self.nominal_columns}")
        
        ################# Pandas Code ####################
        # df_copy = df.copy()

        # for column in self.nominal_columns:
        #     if column not in df_copy.columns:
        #         logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
        #         continue

        #     try:
        #         logger.info(f"Encoding column: {column}")
        #         df_dummies = pd.get_dummies(df_copy[column], prefix=column).astype(int)
        #         df_copy = pd.concat([df_copy, df_dummies], axis=1)
        #         del df_copy[column]
        #         logger.info(f"Column '{column}' encoded successfully.")
        #     except Exception as e:
        #         logger.error(f"Error encoding column '{column}': {e}")
        #         raise

        # logger.info(f"Nominal encoding completed. DataFrame shape: {df_copy.shape}")
        # return df_copy

        ################# Pyspark Code ####################
        for column in self.nominal_columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
                continue

            try:
                logger.info(f"Encoding column: {column}")
                
                # Get distinct values to create dummy columns
                # NOTE: This approach mimics pd.get_dummies but requires collecting values to driver.
                # Suitable for low-cardinality nominal features (e.g. Gender, PaymentMethod).
                categories_row = df.select(column).distinct().collect()
                categories = [row[column] for row in categories_row if row[column] is not None]
                
                for cat in categories:
                    # Create binary column: column_CatName
                    # Sanitize category name for column (basic removal of spaces if any, though raw cat is used in match)
                    safe_cat = str(cat).replace(" ", "_").replace("-", "_") # Simple sanitation
                    # Match pd.get_dummies formatting roughly
                    dummy_col_name = f"{column}_{cat}" 
                    
                    df = df.withColumn(
                        dummy_col_name,
                        when(col(column) == cat, 1).otherwise(0)
                    )
                
                # Drop original column
                df = df.drop(column)
                logger.info(f"Column '{column}' encoded successfully.")
                
            except Exception as e:
                logger.error(f"Error encoding column '{column}': {e}")
                raise

        logger.info(f"Nominal encoding completed. Columns: {len(df.columns)}")
        return df


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_columns, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.ordinal_columns = ordinal_columns
        # self.encoder = OrdinalEncoder()
        self.encoder_model = None

    def encode(self, df: DataFrame) -> DataFrame:
        logger.info(f"Starting ordinal encoding for columns: {self.ordinal_columns}")
        
        ################# Pandas Code ####################
        # df_copy = df.copy()

        # try:
        #     df_copy[self.ordinal_columns] = self.encoder.fit_transform(df_copy[self.ordinal_columns])
        #     logger.info(f"Ordinal encoding completed. DataFrame shape: {df_copy.shape}")
        # except Exception as e:
        #     logger.error(f"Error during ordinal encoding: {e}")
        #     raise

        # return df_copy

        ################# Pyspark Code ####################
        try:
            # Check which columns exist
            valid_cols = [c for c in self.ordinal_columns if c in df.columns]
            if not valid_cols:
                logger.warning("No valid ordinal columns found to encode.")
                return df

            # Prepare temporary output columns to avoid "column already exists" error
            output_cols = [c + "_encoded" for c in valid_cols]

            # StringIndexer in Spark maps strings to indices.
            # stringOrderType="alphabetAsc" mimics sklearn's default lexicographical sort.
            # We use handleInvalid="keep" or "error". default is error.
            indexer = StringIndexer(
                inputCols=valid_cols, 
                outputCols=output_cols, 
                stringOrderType="alphabetAsc",
                handleInvalid="skip" # or 'keep', 'error'
            )
            
            self.encoder_model = indexer.fit(df)
            df_encoded = self.encoder_model.transform(df)

            # Swap back to original names
            for original, encoded in zip(valid_cols, output_cols):
                df_encoded = df_encoded.drop(original).withColumnRenamed(encoded, original)
            
            # StringIndexer returns doubles, we might want integer types but double is fine for ML
            logger.info(f"Ordinal encoding completed. Columns: {len(df_encoded.columns)}")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error during ordinal encoding: {e}")
            raise

    def save_encoder(self, path):
        ################# Pandas Code ####################
        # try:
        #     joblib.dump(self.encoder, path)
        #     logger.info(f"Ordinal encoder saved to {path}")
        # except Exception as e:
        #     logger.error(f"Failed to save ordinal encoder: {e}")
        #     raise
        
        ################# Pyspark Code ####################
        try:
             # Spark models are saved as directories, not single files.
             # If path has an extension, strip it or append logic. 
             # We'll just use the path as provided, assuming user handles directory nature.
            if self.encoder_model:
                self.encoder_model.write().overwrite().save(path)
                logger.info(f"Ordinal encoder (StringIndexerModel) saved to {path}")
            else:
                logger.warning("No ordinal encoder model to save.")
        except Exception as e:
             logger.error(f"Failed to save ordinal encoder: {e}")
             raise

    def load_encoder(self, path):
        ################# Pandas Code ####################
        # try:
        #     self.encoder = joblib.load(path)
        #     logger.info(f"Ordinal encoder loaded from {path}")
        # except Exception as e:
        #     logger.error(f"Failed to load ordinal encoder: {e}")
        #     raise

        ################# Pyspark Code ####################
        try:
            self.encoder_model = StringIndexerModel.load(path)
            logger.info(f"Ordinal encoder (StringIndexerModel) loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load ordinal encoder: {e}")
            raise


class LabelEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, target_column, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.target_column = target_column
        # self.encoder = LabelEncoder()
        self.encoder_model = None

    def encode(self, df: DataFrame) -> DataFrame:
        logger.info(f"Starting label encoding for target column: {self.target_column}")
        
        ################# Pandas Code ####################
        # df_copy = df.copy()

        # try:
        #     df_copy[self.target_column] = self.encoder.fit_transform(df_copy[self.target_column])
        #     logger.info(f"Label encoding completed for column: {self.target_column}")
        # except Exception as e:
        #     logger.error(f"Error during label encoding for column '{self.target_column}': {e}")
        #     raise

        # return df_copy

        ################# Pyspark Code ####################
        try:
            if self.target_column not in df.columns:
                 logger.warning(f"Target column {self.target_column} not found.")
                 return df

            temp_output_column = self.target_column + "_encoded"

            # Label Encoding is typically just String Indexing for the target
            indexer = StringIndexer(
                inputCol=self.target_column, 
                outputCol=temp_output_column,
                stringOrderType="alphabetAsc" 
            )
            
            self.encoder_model = indexer.fit(df)
            df_encoded = self.encoder_model.transform(df)
            
            # Replace original column
            df_encoded = df_encoded.drop(self.target_column).withColumnRenamed(temp_output_column, self.target_column)

            logger.info(f"Label encoding completed for column: {self.target_column}")
            return df_encoded

        except Exception as e:
            logger.error(f"Error during label encoding for column '{self.target_column}': {e}")
            raise

    def save_encoder(self, path):
        ################# Pandas Code ####################
        # try:
        #     joblib.dump(self.encoder, path)
        #     logger.info(f"Label encoder saved to {path}")
        # except Exception as e:
        #     logger.error(f"Failed to save label encoder: {e}")
        #     raise

        ################# Pyspark Code ####################
        try:
            if self.encoder_model:
                self.encoder_model.write().overwrite().save(path)
                logger.info(f"Label encoder (StringIndexerModel) saved to {path}")
            else:
                logger.warning("No label encoder model to save.")
        except Exception as e:
            logger.error(f"Failed to save label encoder: {e}")
            raise

    def load_encoder(self, path):
        ################# Pandas Code ####################
        # try:
        #     self.encoder = joblib.load(path)
        #     logger.info(f"Label encoder loaded from {path}")
        # except Exception as e:
        #     logger.error(f"Failed to load label encoder: {e}")
        #     raise

        ################# Pyspark Code ####################
        try:
            self.encoder_model = StringIndexerModel.load(path)
            logger.info(f"Label encoder (StringIndexerModel) loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load label encoder: {e}")
            raise
