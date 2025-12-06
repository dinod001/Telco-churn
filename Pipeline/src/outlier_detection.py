import pandas as pd
import logging
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OutlierDetectionStrategy(ABC):

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass


class IQROutlierDetection(OutlierDetectionStrategy):

    def detect_outliers(self, df, columns):
        logging.info("Starting IQR outlier detection...")
        outliers = pd.DataFrame(False, index=df.index, columns=columns)

        for col in columns:
            logging.info(f"Processing column: {col}")

            df[col] = df[col].astype(float)

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            logging.info(
                f"[{col}] Q1={Q1}, Q3={Q3}, IQR={IQR}, "
                f"lower={lower_bound}, upper={upper_bound}"
            )

            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

            logging.info(f"[{col}] Outliers detected: {outliers[col].sum()}")

        logging.info("Completed IQR outlier detection.")
        return outliers


class OutlierDetector:

    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect_outliers(self, df, selected_columns):
        logging.info("Running outlier detection strategy...")
        return self.strategy.detect_outliers(df, selected_columns)

    def handle_outliers(self, df, selected_columns):
        logging.info("Handling outliers...")

        outliers = self.detect_outliers(df, selected_columns)
        outliers_count_per_row = outliers.sum(axis=1)

        logging.info("Outliers per row calculated.")

        rows_to_remove = outliers_count_per_row >= 2
        removed_count = rows_to_remove.sum()

        logging.info(f"Rows with >=2 outliers: {removed_count}")

        cleaned_df = df[~rows_to_remove]

        logging.info(
            f"Original rows: {len(df)}, "
            f"Removed rows: {removed_count}, "
            f"Remaining rows: {len(cleaned_df)}"
        )

        return cleaned_df