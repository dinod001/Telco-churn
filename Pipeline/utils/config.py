import os
import yaml
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')


def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f'Error loading configuration: {e}')
        return {}


def get_data_paths():
    config = load_config()
    return config.get('data_paths', {})


def get_columns():
    config = load_config()
    return config.get('columns', {})


def get_missing_values_config():
    config = load_config()
    return config.get('missing_values', {})


def get_outlier_config():
    config = load_config()
    return config.get('outlier_detection', {})


def get_binning_config():
    config = load_config()
    return config.get('feature_binning', {})


def get_encoding_config():
    config = load_config()
    return config.get('feature_encoding', {})


def get_scaling_config():
    config = load_config()
    return config.get('feature_scaling', {})


def get_splitting_config():
    config = load_config()
    return config.get('data_splitting', {})


def get_training_config():
    config = load_config()
    return config.get('training', {})


def get_model_config():
    config = load_config()
    return config.get('model', {})


def get_evaluation_config():
    config = load_config()
    return config.get('evaluation', {})


def get_deployment_config():
    config = load_config()
    return config.get('deployment', {})


def get_logging_config():
    config = load_config()
    return config.get('logging', {})


def get_environment_config():
    config = load_config()
    return config.get('environment', {})


def get_pipeline_config():
    config = load_config()
    return config.get('pipeline', {})


def get_inference_config():
    config = load_config()
    return config.get('inference', {})


def get_mlflow_config():
    config = load_config()
    return config.get('mlflow', {})


def get_config() -> Dict[str, Any]:
    return load_config()


def get_selected_model_config() -> Dict[str, Any]:
    training_config = get_training_config()
    model_config = get_model_config()
    
    selected_model = training_config.get('default_model_type', 'random_forest')
    model_types = model_config.get('model_types', {})
    
    return {
        'model_type': selected_model,
        'model_config': model_types.get(selected_model, {}),
        'training_strategy': training_config.get('default_training_strategy', 'cv'),
        'cv_folds': training_config.get('cv_folds', 5),
        'random_state': training_config.get('random_state', 42)
    }


def get_available_models() -> List[str]:
    model_config = get_model_config()
    return list(model_config.get('model_types', {}).keys())


def update_config(updates: Dict[str, Any]) -> None:
    config_path = CONFIG_FILE
    config = get_config()
    for key, value in updates.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def create_default_config() -> None:
    config_path = CONFIG_FILE
    if not os.path.exists(config_path):
        default_config = {
            "data_paths": {
                "raw_data": "data/raw/Telco-Customer-Churn.csv",
                "processed_data": "data/processed/Telco_Customer_Churn_cleaned.csv",
                "imputed_data": "data/processed/Telco_imputed.csv",
                "processed_dir": "data/processed",
                "artifacts_dir": "artifacts",
                "data_artifacts_dir": "artifacts/data",
                "model_artifacts_dir": "artifacts/models",
                "X_train": "artifacts/data/X_train.csv",
                "X_test": "artifacts/data/X_test.csv",
                "Y_train": "artifacts/data/Y_train.csv",
                "Y_test": "artifacts/data/Y_test.csv"
            },
            "columns": {
                "ordinal_features": ["tenure_bins"],
                "nominal_features": ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"],
                "numerical_features": ["MonthlyCharges", "TotalCharges"],
                "remainder_features": ["SeniorCitizen"],
                "target_feature": "Churn",
                "feature_columns": [
                    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
                ]
            },
            "missing_values": {
                "strategy": "drop",
                "methods": {}
            },
            "outlier_detection": {
                "detection_method": "iqr",
                "handling_method": "remove",
                "z_score_threshold": 3.0
            },
            "feature_binning": {
                "tenure_bins": {
                    "Newer": [0, 12],
                    "Medium": [12, 48],
                    "Long-term": [48, 72]
                },
                "tenure_mapping": {
                    "Newer": 0,
                    "Medium": 1,
                    "Long-term": 2
                }
            },
            "feature_encoding": {
                "nominal_columns": ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"],
                "ordinal_mappings": {
                    "TenureBins": {
                        "Newer": 0,
                        "Medium": 1,
                        "Long-term": 2
                    }
                }
            },
            "feature_scaling": {
                "scaling_type": "minmax",
                "columns_to_scale": ["MonthlyCharges", "TotalCharges"]
            },
            "data_splitting": {
                "split_type": "simple",
                "test_size": 0.2,
                "random_state": 42,
                "n_splits": 5
            },
            "training": {
                "default_model_type": "random_forest",
                "default_training_strategy": "cv",
                "cv_folds": 5,
                "random_state": 42,
                "test_size": 0.2,
                "validation_split": 0.2,
                "early_stopping_patience": 10,
                "max_iterations": 1000,
                "hyperparameter_tuning": {
                    "enabled": False,
                    "search_method": "grid",
                    "cv_folds": 5,
                    "n_iter": 20
                }
            },
            "model": {
                "model_type": "random_forest",
                "training_strategy": "cv",
                "data_path": "data/raw/Telco-Customer-Churn.csv",
                "model_path": "artifacts/models/random_forest_cv_model.pkl",
                "evaluation_path": "artifacts/evaluation/random_forest_cv_evaluation_report.txt",
                "model_params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "random_state": [42]
                },
                "model_types": {
                    "random_forest": {
                        "n_estimators": [100, 200],
                        "max_depth": [10, 15, None],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2],
                        "random_state": 42
                    },
                    "xgboost": {
                        "n_estimators": [100, 200],
                        "max_depth": [5, 10],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0],
                        "random_state": 42
                    },
                    "gradient_boosting": {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "random_state": 42
                    },
                    "logistic_regression": {
                        "random_state": 42,
                        "max_iter": 1000
                    },
                    "svm": {
                        "random_state": 42,
                        "probability": True
                    }
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1-score"],
                "cv_folds": 5,
                "random_state": 42
            },
            "deployment": {
                "model_name": "churn_analysis_model",
                "model_version": "1.0.0",
                "api_endpoint": "/predict",
                "batch_size": 1000
            },
            "inference": {
                "model_name": "random_forest_cv_model",
                "data_path": "artifacts/data/X_test.csv",
                "sample_size": 100,
                "save_path": "artifacts/predictions/predictions.csv",
                "batch_size": 1000,
                "return_proba": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "file": "pipeline.log"
            },
            "mlflow": {
                "tracking_uri": "file:./mlruns",
                "experiment_name": "Churn Analysis",
                "model_registry_name": "churn_prediction",
                "artifact_path": "model",
                "run_name_prefix": "churn_run",
                "tags": {
                    "project": "customer_churn_prediction",
                    "team": "ml_engineering",
                    "environment": "development"
                },
                "autolog": True
            },
            "environment": {
                "experiment_name": "churn_analysis"
            },
            "pipeline": {
                "data_pipeline_name": "data_processing_pipeline",
                "training_pipeline_name": "model_training_pipeline",
                "deployment_pipeline_name": "model_deployment_pipeline",
                "inference_pipeline_name": "inference_pipeline",
                "enable_cache": False
            }
        }
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
        logger.info(f'Created default configuration file: {config_path}')


create_default_config()