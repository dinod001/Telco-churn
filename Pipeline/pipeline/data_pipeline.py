import os
import sys
import logging
import pandas as pd
from typing import Dict
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import ReadCSV
from handling_missing_values import DropMissingValuesStrategy
from outlier_detection import OutlierDetector,IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import NominalEncodingStrategy,OrdinalEncodingStrategy,LabelEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config

def data_pipeline(
    data_path: str = 'data/raw/Telco-Customer-Churn.csv',
    )->Dict[str, np.ndarray]:

    data_paths=get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scalling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    print("step 01 data ingestion")

    artifacts_dir = os.path.join(os.path.dirname(__file__),'..',data_paths['data_artifacts_dir'])
    encoder_dir = os.path.join(os.path.dirname(__file__),'..',data_paths['encoder_dir'])
    os.makedirs(encoder_dir, exist_ok=True)

    X_train_path = os.path.join(artifacts_dir,'X_train.csv')
    X_test_path = os.path.join(artifacts_dir,'X_test.csv')
    Y_train_path = os.path.join(artifacts_dir,'Y_train.csv')
    Y_test_path = os.path.join(artifacts_dir,'Y_test.csv')

    if os.path.exists(X_train_path) and \
       os.path.exists(X_test_path) and \
       os.path.exists(Y_train_path) and \
       os.path.exists(Y_test_path):

       X_train = pd.read_csv(X_train_path)
       X_test = pd.read_csv(X_test_path)
       Y_train = pd.read_csv(Y_train_path)
       Y_test = pd.read_csv(Y_test_path)
    
    os.makedirs(data_paths['data_artifacts_dir'],exist_ok=True)

    ingestor = ReadCSV(data_path)
    df = ingestor.ingest()

    print("02. Handling missing values")
    missing_value_handler = DropMissingValuesStrategy(columns['critical_column'])
    df = missing_value_handler.handle(df)

    print("03 Handling outliers")
    outlier_handler = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_handler.handle_outliers(df,columns['numerical_features'])

    print("04. Feature binning")
    binning_handler = CustomBinningStrategy()
    df = binning_handler.bin_feature(df,binning_config['binning_column'])

    print("05. Feature encoding")
    nominal_encoding_handler = NominalEncodingStrategy(encoding_config['nominal_columns'])
    df = nominal_encoding_handler.encode(df)

    ordinal_encoding_handler = OrdinalEncodingStrategy(columns['ordinal_features'])
    df = ordinal_encoding_handler.encode(df)
    ordinal_encoding_handler.save_encoder(os.path.join(encoder_dir,'ordinal_encoder.joblib'))

    label_encoding_handler = LabelEncodingStrategy(columns['target_feature'])
    df = label_encoding_handler.encode(df)
    label_encoding_handler.save_encoder(os.path.join(encoder_dir,'label_encoder.joblib'))

    print("06. feature scalling")
    scaller = MinMaxScalingStrategy(scaler_path=os.path.join(encoder_dir,'minmax_scaler.joblib'))
    df = scaller.scale(df,columns['numerical_features'])

    df = df.drop(columns=['customerID'])

    print("07. Data splitting")
    spliter = SimpleTrainTestSplitStratergy(apply_smote=True)
    X_train, X_test, Y_train, Y_test = spliter.split_data(df,columns['target_feature'])

    X_train.to_csv(X_train_path,index=False)
    X_test.to_csv(X_test_path,index=False)
    Y_train.to_csv(Y_train_path,index=False)
    Y_test.to_csv(Y_test_path,index=False)

    print(f"X train shape {X_train.shape}")
    print(f"X test shape {X_test.shape}")
    print(f"Y train shape {Y_train.shape}")
    print(f"Y test shape {Y_test.shape}")

data_pipeline()