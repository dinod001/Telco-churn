"""
Optimized Model Inference for Streaming/Real-time Predictions

This version extracts transformation parameters from Spark models
and applies them directly in Pandas for low-latency inference.
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np
import sys, os
from abc import ABC, abstractmethod
import joblib
import json

# ---------------------- LOGGING CONFIGURATION -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from config import (
    get_columns,
    get_binning_config,
    get_encoding_config,
    get_scaling_config
)

# ---------------------- LOAD CONFIGS -------------------------------
columns = get_columns()
binning_config = get_binning_config()
encoding_config = get_encoding_config()
scaling_config = get_scaling_config()

# ---------------------- LOAD ENCODERS & MODEL ----------------------
encoders_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "encoders")
model_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "models")

# Load ML model
model = joblib.load(os.path.join(model_path, "Telco-Customer-Churn.joblib"))

# ---------------------- EXTRACT SPARK MODEL PARAMETERS ----------------------
# For streaming inference, we extract parameters once and apply them quickly

def load_spark_model_params():
    """Extract transformation parameters from Spark models for fast Pandas inference"""
    from pyspark.ml.feature import StringIndexerModel, MinMaxScalerModel
    from spark_session import get_or_create_spark_session
    
    # Initialize SparkSession FIRST
    spark = get_or_create_spark_session()
    
    # Load Spark models
    ordinal_encoder_model = StringIndexerModel.load(os.path.join(encoders_dir, "ordinal_encoder_model"))
    label_encoder_model = StringIndexerModel.load(os.path.join(encoders_dir, "label_encoder_model"))
    scaler_model = MinMaxScalerModel.load(os.path.join(encoders_dir, "minmax_scaler_model"))
    
    # Extract ordinal encoding mappings
    ordinal_mappings = {}
    for i, col in enumerate(columns["ordinal_features"]):
        if i < len(ordinal_encoder_model.labelsArray):
            labels = ordinal_encoder_model.labelsArray[i]
            ordinal_mappings[col] = {label: float(idx) for idx, label in enumerate(labels)}
    
    # Extract label encoding mapping
    label_mappings = {label: float(idx) for idx, label in enumerate(label_encoder_model.labels)}
    label_inverse = {idx: label for label, idx in label_mappings.items()}
    
    # Extract scaling parameters
    scaler_info = scaler_model.originalMin
    scaler_max = scaler_model.originalMax
    
    # Create min-max mapping for numerical features
    scaling_params = {}
    for i, col in enumerate(columns["numerical_features"]):
        if i < len(scaler_info):
            scaling_params[col] = {
                'min': scaler_info[i],
                'max': scaler_max[i]
            }
    
    return ordinal_mappings, label_inverse, scaling_params

# Load parameters once at startup
logger.info("Loading Spark model parameters for fast inference...")
ordinal_mappings, label_inverse_map, scaling_params = load_spark_model_params()
logger.info("Parameters loaded successfully!")

# ===================================================================

def tenure_binning(tenure):
    """Fast tenure binning"""
    if tenure <= 12:
        return "Newer"
    elif tenure <= 48:
        return "Medium"
    else:
        return "Long-term"

def apply_ordinal_encoding_fast(df, mappings):
    """Apply ordinal encoding using pre-loaded mappings"""
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

def apply_minmax_scaling_fast(df, params):
    """Apply min-max scaling using pre-loaded parameters"""
    for col, scale_info in params.items():
        if col in df.columns:
            min_val = scale_info['min']
            max_val = scale_info['max']
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
    return df

# ===================================================================

class ModelInference(ABC):
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Returns prediction"""
        raise NotImplementedError


class Preprocessing(ModelInference):

    def __init__(self, data: Dict):
        self.data = data

    # ---------------------- FAST PREPROCESSING ----------------------
    def data_preprocessing(self) -> pd.DataFrame:
        """Optimized preprocessing without Spark overhead"""
        df = pd.DataFrame([self.data])

        # 1. Feature Binning (Pure Python - FAST)
        if binning_config["binning_column"] in df.columns:
            new_col = f"{binning_config['binning_column']}_bins"
            df[new_col] = df[binning_config["binning_column"]].apply(tenure_binning)
            df = df.drop(columns=[binning_config["binning_column"]])

        # 2. Nominal Encoding (Pandas get_dummies - FAST)
        for column in encoding_config["nominal_columns"]:
            if column in df.columns:
                dummies = pd.get_dummies(df[column], prefix=column).astype(int)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[column])

        # 3. Ordinal Encoding (Direct mapping - FAST)
        df = apply_ordinal_encoding_fast(df, ordinal_mappings)

        # 4. Scaling (Direct formula - FAST)
        df = apply_minmax_scaling_fast(df, scaling_params)

        # 5. Drop Unnecessary Column
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # 6. Ensure all columns exist and match model's training order
        expected_columns = [
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 
            'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
            'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
            'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_No', 
            'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
            'InternetService_DSL', 'OnlineSecurity_No', 'OnlineSecurity_Yes', 
            'OnlineSecurity_No internet service', 'OnlineBackup_No', 'OnlineBackup_Yes', 
            'OnlineBackup_No internet service', 'DeviceProtection_No', 'DeviceProtection_Yes', 
            'DeviceProtection_No internet service', 'TechSupport_No', 'TechSupport_Yes', 
            'TechSupport_No internet service', 'StreamingTV_No', 'StreamingTV_Yes', 
            'StreamingTV_No internet service', 'StreamingMovies_No', 'StreamingMovies_Yes', 
            'StreamingMovies_No internet service', 'Contract_Month-to-month', 'Contract_One year',
            'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Mailed check',
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Electronic check',
            'tenure_bins'
        ]

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]
        return df

    # ---------------------- PREDICT METHOD ----------------------
    def predict(self):
        processed_df = self.data_preprocessing()
        prediction = model.predict(processed_df)
        prediction_proba = model.predict_proba(processed_df)

        # Use pre-loaded inverse mapping
        pred_label = label_inverse_map.get(int(prediction[0]), "Unknown")

        return {
            "prediction": pred_label,
            "probability": str(round(float(max(prediction_proba[0]) * 100), 2)) + "%"
        }
