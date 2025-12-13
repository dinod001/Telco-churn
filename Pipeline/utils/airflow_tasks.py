import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# =========================
# Configure logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# Project Environment Setup
# =========================
def setup_project_environment() -> str:
    """
    Setup the project environment by adding key project directories to sys.path
    and setting PYTHONPATH. Returns the absolute path to the project root.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    paths_to_add = [
        str(project_root),
        str(project_root / 'src'),
        str(project_root / 'utils'),
        str(project_root / 'pipelines')
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    os.environ['PYTHONPATH'] = ':'.join(paths_to_add)
    logger.info(f"Project environment configured. PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    return str(project_root)

# =========================
# Input Data Validation
# =========================
def validate_input_data(data_path: str = 'data/raw') -> Dict[str,Any]:
    """
    Validate raw input data existence and file size.
    Returns a summary dict indicating success or warnings.
    """
    project_root = setup_project_environment()
    full_path = Path(project_root) / data_path

    logger.info(f"Validating input data at: {full_path}")

    if not full_path.exists():
        logger.warning(f"Input data file not found: {full_path}")
        return {
            'status': 'warning',
            'message': 'Input data file not found',
            'file_path': str(full_path)
        }
    
    file_size = full_path.stat().st_size
    if file_size == 0:
        logger.warning(f"Input data file is empty: {full_path}")
        return {
            'status': 'warning',
            'message': 'Input data file is empty',
            'file_path': str(full_path)
        }
    
    logger.info(f"✅ Input data validation passed: {file_size} bytes")
    return {
        'status': 'success',
        'file_path': str(full_path),
        'file_size_bytes': file_size,
        'message': 'Input data file exists and has content'
    }

# =========================
# Processed Data Validation
# =========================
def validate_processed_data(data_path: str = 'artifacts/data/X_train.csv') -> Dict[str,Any]:
    """
    Validate processed data existence and file size.
    """
    project_root = setup_project_environment()
    full_path = Path(project_root) / data_path

    logger.info(f"Validating processed data at: {full_path}")

    if not full_path.exists():
        logger.warning(f"Processed data file not found: {full_path}")
        return {
            'status': 'warning',
            'message': 'Processed data file not found',
            'file_path': str(full_path)
        }

    file_size = full_path.stat().st_size
    if file_size == 0:
        logger.warning(f"Processed data file is empty: {full_path}")
        return {
            'status': 'warning',
            'message': 'Processed data file is empty',
            'file_path': str(full_path)
        }

    logger.info(f"✅ Processed data validation passed: {file_size} bytes")
    return {
        'status': 'success',
        'file_path': str(full_path),
        'file_size_bytes': file_size,
        'message': 'Processed data file exists and has content'
    }

# =========================
# Model Validation
# =========================
def validate_trained_model(model_path: str = 'artifacts/models') -> Dict[str,Any]:
    """
    Validate that trained model files exist.
    """
    project_root = setup_project_environment()
    model_dir = Path(project_root) / model_path

    logger.info(f"Validating trained model at: {model_dir}")

    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return {
            'status': 'warning',
            'message': 'Model directory not found. Run training pipeline first.',
            'model_directory': str(model_dir)
        }

    model_files = list(model_dir.glob('**/*'))
    if not model_files:
        logger.warning(f"No model files found in: {model_dir}")
        return {
            'status': 'warning',
            'message': 'No model files found. Run training pipeline first.',
            'model_directory': str(model_dir)
        }

    logger.info(f"✅ Model validation passed: {len(model_files)} file(s) found")
    return {
        'status': 'success',
        'model_directory': str(model_dir),
        'model_files_count': len(model_files),
        'message': 'Model files found'
    }

# =========================
# Data Pipeline Execution
# =========================
def run_data_pipeline(data_path: str = "data/raw/Telco-Customer-Churn.csv") -> Dict[str, Any]:
    """
    Run the data pipeline for preprocessing and return summary info.
    """
    project_root = setup_project_environment()
    try:
        os.chdir(project_root)
        from data_pipeline import data_pipeline

        logger.info(f"Starting data pipeline: {data_path}")
        result = data_pipeline(data_path=data_path)
        
        logger.info("✓ Data pipeline completed successfully")
        return {
            'status': 'success',
            'X_train_shape': result['X_train'].shape if 'X_train' in result else None,
            'X_test_shape': result['X_test'].shape if 'X_test' in result else None,
            'Y_train_shape': result['Y_train'].shape if 'Y_train' in result else None,
            'Y_test_shape': result['Y_test'].shape if 'Y_test' in result else None,
            'message': 'Data pipeline completed successfully'
        }
    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        raise

# =========================
# Training Pipeline Execution
# =========================
def run_training_pipeline(
        data_path: str = 'data/raw/ChurnModelling.csv',
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        model_path: str = 'artifacts/models/Telco-Customer-Churn.joblib'
    ) -> Dict[str, Any]:
    """
    Run model training pipeline and save trained model.
    """
    project_root = setup_project_environment()
    try:
        os.chdir(project_root)
        from training_pipeline import training_pipeline

        training_pipeline(
            data_path=data_path,
            model_params=model_params,
            test_size=test_size,
            random_state=random_state,
            model_path=model_path
        )

        logger.info("✓ Training pipeline completed successfully")
        return {
            'status': 'success',
            'model_path': model_path,
            'message': 'Training pipeline completed successfully'
        }
    except Exception as e:
        logger.error(f"✗ Training pipeline failed: {str(e)}")
        raise

# =========================
# Inference Pipeline Execution
# =========================
def run_inference_pipeline(sample_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run inference on sample data using trained model.
    """
    project_root = setup_project_environment()
    try:
        os.chdir(project_root)
        from model_inference_pipeline import predict

        if sample_data is None:
            # Example sample data for inference
            sample_data = {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.10,
                "TotalCharges": 1085.50
            }

        logger.info("Starting inference pipeline")
        prediction = predict(sample_data)
        logger.info("✓ Inference pipeline completed successfully")
        return {
            'status': 'success',
            'prediction': prediction,
            'message': 'Inference pipeline completed successfully'
        }
    except Exception as e:
        logger.error(f"✗ Inference pipeline failed: {str(e)}")
        raise
