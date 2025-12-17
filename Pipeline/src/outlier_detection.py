import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when
from spark_session import get_or_create_spark_session
from functools import reduce

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OutlierDetectionStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def detect_outliers(self, df: DataFrame, columns: list) -> DataFrame:
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    def __init__(self, spark: Optional[SparkSession] = None):
        super().__init__(spark)

    def detect_outliers(self, df: DataFrame, columns: list) -> DataFrame:
        logger.info("Starting IQR outlier detection...")
        
        ################# Pandas Code ####################
        # outliers = pd.DataFrame(False, index=df.index, columns=columns)

        # for col in columns:
        #     logging.info(f"Processing column: {col}")

        #     df[col] = df[col].astype(float)

        #     Q1 = df[col].quantile(0.25)
        #     Q3 = df[col].quantile(0.75)
        #     IQR = Q3 - Q1

        #     lower_bound = Q1 - 1.5 * IQR
        #     upper_bound = Q3 + 1.5 * IQR

        #     logging.info(
        #         f"[{col}] Q1={Q1}, Q3={Q3}, IQR={IQR}, "
        #         f"lower={lower_bound}, upper={upper_bound}"
        #     )

        #     outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        #     logging.info(f"[{col}] Outliers detected: {outliers[col].sum()}")

        # logging.info("Completed IQR outlier detection.")
        # return outliers
        
        ################# Pyspark Code ####################
        outlier_flags = []

        for column in columns:
            logger.info(f"Processing column: {column}")

            # Cast to double to ensure numerical operations like approxQuantile work
            # This mirrors df[col] = df[col].astype(float)
            df = df.withColumn(column, col(column).cast("double"))

            # Calculate Quantiles and IQR
            # approxQuantile is more efficient for large datasets
            quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
            Q1 = quantiles[0]
            Q3 = quantiles[1]
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            logger.info(
                f"[{column}] Q1={Q1}, Q3={Q3}, IQR={IQR}, "
                f"lower={lower_bound}, upper={upper_bound}"
            )

            # Create an expression that returns 1 if outlier, 0 otherwise
            is_outlier = (col(column) < lower_bound) | (col(column) > upper_bound)
            outlier_flags.append(when(is_outlier, 1).otherwise(0))

        # Sum the outlier flags per row
        if outlier_flags:
            total_outliers = reduce(lambda x, y: x + y, outlier_flags)
            df_with_count = df.withColumn("outlier_count", total_outliers)
        else:
            df_with_count = df.withColumn("outlier_count", 0) # Fallback if no columns selected

        logger.info("Completed IQR outlier calculation.")
        return df_with_count


class OutlierDetector:

    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect_outliers(self, df, selected_columns):
        logging.info("Running outlier detection strategy...")
        return self.strategy.detect_outliers(df, selected_columns)

    def handle_outliers(self, df, selected_columns):
        logging.info("Handling outliers...")

        ################# Pandas Code ####################
        # outliers = self.detect_outliers(df, selected_columns)
        # outliers_count_per_row = outliers.sum(axis=1)

        # logging.info("Outliers per row calculated.")

        # rows_to_remove = outliers_count_per_row >= 2
        # removed_count = rows_to_remove.sum()

        # logging.info(f"Rows with >=2 outliers: {removed_count}")

        # cleaned_df = df[~rows_to_remove]

        # logging.info(
        #     f"Original rows: {len(df)}, "
        #     f"Removed rows: {removed_count}, "
        #     f"Remaining rows: {len(cleaned_df)}"
        # )

        # return cleaned_df

        ################# Pyspark Code ####################
        # Detect outliers returns the dataframe with 'outlier_count' column
        df_with_counts = self.detect_outliers(df, selected_columns)
        
        initial_rows = df_with_counts.count()
        
        # Filter rows where outlier_count < 2
        cleaned_df = df_with_counts.filter(col("outlier_count") < 2)
        
        final_rows = cleaned_df.count()
        removed_count = initial_rows - final_rows
        
        logging.info(f"Rows with >=2 outliers: {removed_count}")

        # Drop the helper column
        cleaned_df = cleaned_df.drop("outlier_count")

        logging.info(
            f"Original rows: {initial_rows}, "
            f"Removed rows: {removed_count}, "
            f"Remaining rows: {final_rows}"
        )

        return cleaned_df