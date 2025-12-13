import os
import sys
import logging
import pandas as pd
from typing import Dict
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# Configure logging
# ───────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Imports from project structure
# ───────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_ingestion import ReadCSV
from handling_missing_values import DropMissingValuesStrategy
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import (
    NominalEncodingStrategy, 
    OrdinalEncodingStrategy,
    LabelEncodingStrategy
)
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleTrainTestSplitStratergy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from config import (
    get_data_paths, 
    get_columns, 
    get_outlier_config, 
    get_binning_config, 
    get_encoding_config, 
    get_scaling_config, 
    get_splitting_config
)
from mlflow_utils import MLflowTracker

# ───────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ───────────────────────────────────────────────────────────────────────────────
def data_pipeline(
    data_path: str = "data/raw/Telco-Customer-Churn.csv"
) -> Dict[str, np.ndarray]:

    logger.info("\n\n✨ Initializing configuration...\n\n")

    # Load configurations
    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    # Mlflow setup
    mlflow_tracker = MLflowTracker()
    mlflow_tracker.setup_mlflow_autolog()
    run_tags = mlflow_tracker.create_mlflow_run_tags(
        'data_pipeline',
        {
            'data_path': data_path,
            'columns': columns,
            'outlier_config': outlier_config,
            'binning_config': binning_config,
            'encoding_config': encoding_config,
            'scaling_config': scaling_config,
            'splitting_config': splitting_config
        }
    )

    mlflow_tracker.start_run(run_name="data_pipeline",tags=run_tags)

    # Create directories
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", data_paths["data_artifacts_dir"])
    encoder_dir = os.path.join(os.path.dirname(__file__), "..", data_paths["encoder_dir"])
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(encoder_dir, exist_ok=True)

    # Output paths
    X_train_path = os.path.join(artifacts_dir, "X_train.csv")
    X_test_path  = os.path.join(artifacts_dir, "X_test.csv")
    Y_train_path = os.path.join(artifacts_dir, "Y_train.csv")
    Y_test_path  = os.path.join(artifacts_dir, "Y_test.csv")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 01 — Cached data
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n📦 STEP 01 — Checking cached datasets...\n\n")
    if (
        os.path.exists(X_train_path) and 
        os.path.exists(X_test_path) and 
        os.path.exists(Y_train_path) and 
        os.path.exists(Y_test_path)
    ):
        logger.info("✅ Cached datasets found — loading...")
        return {
            "X_train": pd.read_csv(X_train_path),
            "X_test": pd.read_csv(X_test_path),
            "Y_train": pd.read_csv(Y_train_path),
            "Y_test": pd.read_csv(Y_test_path)
        }

        mlflow_tracker.log_data_pipeline_metrics({
            'total_rows': len(df),
            'train_rows': len(X_train),
            'test_rows': len(X_test),
            'num_features': X_train.shape[1],
            'missing_values': X_train.isna().sum().sum(),
            'outliers_removed': 0 
        })

        mlflow_tracker.end_run()
    logger.info("──────────────────────────────────────────────")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 02 — Data Ingestion
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n🛠 STEP 02 — Data Ingestion\n\n")
    df = ReadCSV(data_path).ingest()
    logger.info(f"📊 Data shape: {df.shape}")
    logger.info("──────────────────────────────────────────────")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 03 — Handling Missing Values
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n💧 STEP 03 — Handling Missing Values\n\n")
    df = DropMissingValuesStrategy(columns["critical_column"]).handle(df)
    logger.info("──────────────────────────────────────────────")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 04 — Outlier Detection
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n🚨 STEP 04 — Outlier Detection\n\n")
    df = OutlierDetector(strategy=IQROutlierDetection()).handle_outliers(
        df, columns["numerical_features"]
    )
    logger.info("──────────────────────────────────────────────")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 05 — Feature Binning
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n📦 STEP 05 — Feature Binning\n\n")
    df = CustomBinningStrategy().bin_feature(df, binning_config["binning_column"])
    logger.info("──────────────────────────────────────────────")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 06 — Feature Encoding
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n🔤 STEP 06 — Feature Encoding\n\n")
    # Nominal
    df = NominalEncodingStrategy(encoding_config["nominal_columns"]).encode(df)
    # Ordinal
    ordinal_encoder = OrdinalEncodingStrategy(columns["ordinal_features"])
    df = ordinal_encoder.encode(df)
    ordinal_encoder.save_encoder(os.path.join(encoder_dir, "ordinal_encoder.joblib"))
    # Label (Target)
    label_encoder = LabelEncodingStrategy(columns["target_feature"])
    df = label_encoder.encode(df)
    label_encoder.save_encoder(os.path.join(encoder_dir, "label_encoder.joblib"))
    logger.info("──────────────────────────────────────────────")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 07 — Feature Scaling
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n📏 STEP 07 — Feature Scaling\n\n")
    scaler = MinMaxScalingStrategy(
        scaler_path=os.path.join(encoder_dir, "minmax_scaler.joblib")
    )
    df = scaler.scale(df, columns["numerical_features"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("🆔 Dropped 'customerID' column")
    logger.info("──────────────────────────────────────────────")

    # ───────────────────────────────────────────────────────────────────────────
    # STEP 08 — Train/Test Split
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n\n✂️ STEP 08 — Data Splitting\n\n")
    splitter = SimpleTrainTestSplitStratergy()
    X_train, X_test, Y_train, Y_test = splitter.split_data(df, columns["target_feature"])
    logger.info("──────────────────────────────────────────────")

    # Save artifacts
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    Y_train.to_csv(Y_train_path, index=False)
    Y_test.to_csv(Y_test_path, index=False)

    logger.info("\n\n🎉 Data pipeline completed successfully!\n\n")
    logger.info(f"✅ X_train shape: {X_train.shape}")
    logger.info(f"✅ X_test shape: {X_test.shape}")
    logger.info(f"✅ Y_train shape: {Y_train.shape}")
    logger.info(f"✅ Y_test shape: {Y_test.shape}\n")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test
    }

# Run the pipeline
if __name__ == "__main__":
    data_pipeline()
