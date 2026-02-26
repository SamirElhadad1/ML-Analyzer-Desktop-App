import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    pass

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numeric_cols:
        if df[col].nunique() == 2:
            categorical_cols.append(col)
            numeric_cols.remove(col)
    return numeric_cols, categorical_cols

def handle_missing_values(df: pd.DataFrame,numeric_cols: List[str],categorical_cols: List[str],strategy: str = 'mean') -> pd.DataFrame:

    try:
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
            return df_clean

        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy=strategy)
            df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])

        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = categorical_imputer.fit_transform(df_clean[categorical_cols])
            
        return df_clean
        
    except Exception as e:
        raise PreprocessingError(f"Error handling missing values: {str(e)}")

def encode_categorical_features(df: pd.DataFrame,categorical_cols: List[str],encoding_method: str = 'label') -> Tuple[pd.DataFrame, Dict]:
    try:
        df_encoded = df.copy()
        encoders = {}

        if encoding_method == 'label':
            for col in categorical_cols:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
                encoders[col] = encoder
        elif encoding_method == 'onehot':
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df_encoded[categorical_cols])
            new_cols = []
            for i, col in enumerate(categorical_cols):
                categories = encoder.categories_[i]
                for cat in categories:
                    new_cols.append(f"{col}_{cat}")
            encoded_df = pd.DataFrame(encoded_data, columns=new_cols)
            df_encoded = df_encoded.drop(columns=categorical_cols)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            
            encoders['onehot'] = encoder
            
        return df_encoded, encoders
        
    except Exception as e:
        raise PreprocessingError(f"Error encoding categorical features: {str(e)}")

def scale_numeric_features(df: pd.DataFrame,
                            numeric_cols: List[str],scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:

    try:
        df_scaled = df.copy()
        
        if not scaler:
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        else:
            df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
            
        return df_scaled, scaler
        
    except Exception as e:
        raise PreprocessingError(f"Error scaling numeric features: {str(e)}")

def preprocess_data(df, target_column, handle_missing='mean', encoding_method='label', scale_numeric=True):
    df = df.copy()
    artifacts = {'encoders': {}, 'scaler': None}

    if handle_missing == 'drop':
        df = df.dropna()
    else:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    fill_value = df[col].mean() if handle_missing == 'mean' else df[col].median()
                    df[col] = df[col].fillna(fill_value)
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    if encoding_method == 'onehot':
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        for col in df.columns:
            if col != target_column and df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                artifacts['encoders'][col] = le

    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        artifacts['encoders'][target_column] = le

    if scale_numeric:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            artifacts['scaler'] = scaler
    return df, artifacts

def inverse_transform_prediction(prediction: Union[int, float, np.ndarray],
                                artifacts: Dict,target_column: str) -> Union[int, float, str]:
    try:
        if target_column in artifacts['categorical_cols']:
            encoder = artifacts['encoders'][target_column]
            return encoder.inverse_transform([prediction])[0]
        else:
            return prediction

    except Exception as e:
        raise PreprocessingError(f"Error in inverse transform: {str(e)}")