import logging
import joblib
from typing import Dict
import pandas as pd
import sys, os
from abc import ABC, abstractmethod

# ---------------------- LOGGING CONFIGURATION -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs from encoding/binning modules
logging.getLogger("feature_binning").setLevel(logging.ERROR)
logging.getLogger("feature_encoding").setLevel(logging.ERROR)

# --------------------------------------------------------------------

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from feature_binning import CustomBinningStrategy
from feature_encoding import NominalEncodingStrategy

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

ordinal_encoder = joblib.load(os.path.join(encoders_dir, "ordinal_encoder.joblib"))
label_encoder = joblib.load(os.path.join(encoders_dir, "label_encoder.joblib"))
minmax_scaler = joblib.load(os.path.join(encoders_dir, "minmax_scaler.joblib"))
model = joblib.load(os.path.join(model_path, "Telco-Customer-Churn.joblib"))

# ===================================================================

class ModelInference(ABC):
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Returns prediction"""
        raise NotImplementedError


class Preprocessing(ModelInference):

    def __init__(self, data: Dict):
        self.data = data

    # ---------------------- PREPROCESSING ----------------------
    def data_preprocessing(self) -> pd.DataFrame:
        df = pd.DataFrame([self.data])

        # 1. Feature Binning
        df = CustomBinningStrategy().bin_feature(df, binning_config["binning_column"])

        # 2. Nominal Encoding
        df = NominalEncodingStrategy(encoding_config["nominal_columns"]).encode(df)

        # 3. Ordinal Encoding
        df[columns["ordinal_features"]] = ordinal_encoder.transform(
            df[columns["ordinal_features"]]
        )

        # 4. Scaling
        df[columns["numerical_features"]] = minmax_scaler.transform(
            df[columns["numerical_features"]]
        )

        # 5. Drop Unnecessary Column
        df = df.drop(columns=["customerID"])

        # 6. Ensure all columns exist
        expected_columns = [
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure_bins',
            'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
            'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
            'PhoneService_Yes', 'MultipleLines_No',
            'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_DSL', 'InternetService_Fiber optic',
            'InternetService_No', 'OnlineSecurity_No',
            'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            'OnlineBackup_No', 'OnlineBackup_No internet service',
            'OnlineBackup_Yes', 'DeviceProtection_No',
            'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
            'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No', 'StreamingMovies_No internet service',
            'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
            'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
            'PaymentMethod_Bank transfer (automatic)',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
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

        return {
            "prediction": label_encoder.inverse_transform(prediction)[0],
            "probability": str(round(float(max(prediction_proba[0]) * 100), 2)) + "%"


        }

