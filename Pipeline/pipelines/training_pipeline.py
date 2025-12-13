import os
import sys
import logging
import pandas as pd
from typing import Dict, Any, Optional

from data_pipeline import data_pipeline
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import XGboostModelBuilder, RandomForestModelBuilder
from config import get_model_config, get_data_paths
from mlflow_utils import MLflowTracker

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------
def training_pipeline(
        data_path: str = 'data/raw/ChurnModelling.csv',
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        model_path: str = 'artifacts/models/Telco-Customer-Churn.joblib'
):
    logger.info("\n\n🚀 Starting Training Pipeline...\n\n")

    data_files = get_data_paths()
    required_files = ["X_train", "X_test", "Y_train", "Y_test"]

    # -----------------------------------------------------------------
    # Check if artifacts exist
    # -----------------------------------------------------------------
    logger.info("\n\n🔍 Checking for cached data artifacts...\n\n")
    if not all(os.path.exists(data_files[file]) for file in required_files):
        logger.warning("📁 Data artifacts not found — running data pipeline...")
        data_pipeline()
    else:
        logger.info("📦 Loading existing data artifacts...")
    
    # Setup Mflow
    mlflow_tracker = MLflowTracker()
    mlflow_tracker.setup_mlflow_autolog()
    run_tags = mlflow_tracker.create_mlflow_run_tags(
        'training_pipeline',
        {   

            'model_type': 'XGBoost',
            'model_params': model_params,
            'test_size': test_size,
            'random_state': random_state,
            'model_path': model_path
        }
    )

    mlflow_tracker.start_run(run_name="training_pipeline",tags=run_tags)

    # -----------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------
    logger.info("\n\n📥 Reading training & test datasets...\n\n")
    X_train = pd.read_csv(data_files["X_train"])
    X_test = pd.read_csv(data_files["X_test"])
    Y_train = pd.read_csv(data_files["Y_train"])
    Y_test = pd.read_csv(data_files["Y_test"])

    # -----------------------------------------------------------------
    # Model Building
    # -----------------------------------------------------------------
    logger.info("\n\n🧱 Building model: XGBoost...\n\n")
    model_builder = XGboostModelBuilder()
    model = model_builder.build_model()

    # -----------------------------------------------------------------
    # Model Training
    # -----------------------------------------------------------------
    logger.info("\n\n🏋️‍♂️ Training model...\n\n")
    trainer = ModelTrainer(param_grid=model_params)

    model, train_score = trainer.train(
        model=model,
        X_train=X_train,
        Y_train=Y_train
    )

    logger.info(f"✅ Training Completed — Score: {train_score:.4f}")
    trainer.save_model(model, model_path)
    logger.info(f"💾 Model saved to: {model_path}")

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    logger.info("\n\n📊 Evaluating model...\n\n")
    evaluator = ModelEvaluator(model, "XGBoost")
    evaluation_results = evaluator.evaluate(X_test, Y_test)

    # Remove confusion matrix before printing
    results_display = evaluation_results.copy()
    results_display.pop("cm", None)
    
    #mlflow
    model_params = get_model_config()['model_params']

    mlflow_tracker.log_training_metrics(model, evaluation_results,model_params)

    mlflow_tracker.end_run()

    logger.info("✨ Evaluation Results:")
    for k, v in results_display.items():
        logger.info(f"   - {k}: {v}")

    logger.info("\n\n🏁 Pipeline completed successfully! 🎉\n\n")


# ---------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model_config = get_model_config()
    model_params = model_config.get("model_params")

    training_pipeline(model_params=model_params)
