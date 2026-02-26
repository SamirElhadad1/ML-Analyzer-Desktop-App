from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Union, Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Custom exception for prediction-related errors."""
    pass

def validate_input(input_data: Union[Dict[str, Any], pd.DataFrame], 
                    feature_columns: List[str]) -> pd.DataFrame:
    """
    Validate input data for prediction.
    
    Args:
        input_data: Dictionary or DataFrame containing feature values
        feature_columns: List of expected feature column names
        
    Returns:
        pd.DataFrame: Validated and formatted input data
        
    Raises:
        PredictionError: If validation fails
    """
    try:
        # Convert dict to DataFrame if necessary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
            
        # Check for missing features
        missing_features = set(feature_columns) - set(input_df.columns)
        if missing_features:
            raise PredictionError(f"Missing features: {missing_features}")
            
        # Check for extra features
        extra_features = set(input_df.columns) - set(feature_columns)
        if extra_features:
            logger.warning(f"Extra features provided: {extra_features}")
            input_df = input_df[feature_columns]
            
        # Check for missing values
        if input_df.isnull().any().any():
            raise PredictionError("Input contains missing values")
            
        # Ensure numeric values
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                raise PredictionError(f"Non-numeric value found in column: {col}")
                
        return input_df
        
    except Exception as e:
        raise PredictionError(f"Input validation failed: {str(e)}")

def preprocess_input(input_df: pd.DataFrame, 
                    scaler: StandardScaler,
                    categorical_columns: List[str] = None) -> np.ndarray:
    """
    Preprocess input data to match training data format.
    
    Args:
        input_df: DataFrame containing input features
        scaler: Fitted StandardScaler instance
        categorical_columns: List of categorical column names
        
    Returns:
        np.ndarray: Preprocessed input data
        
    Raises:
        PredictionError: If preprocessing fails
    """
    try:
        # Handle categorical variables if specified
        if categorical_columns:
            # Create dummy variables for categorical columns
            input_df = pd.get_dummies(input_df, columns=categorical_columns)
            
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        return input_scaled
        
    except Exception as e:
        raise PredictionError(f"Preprocessing failed: {str(e)}")

def make_prediction(model: Any,
                   input_data: Union[Dict[str, Any], pd.DataFrame],
                   feature_columns: List[str],
                   scaler: StandardScaler,
                   task: str = "classification",
                   categorical_columns: List[str] = None) -> Union[int, float, str]:
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained scikit-learn model
        input_data: Dictionary or DataFrame containing feature values
        feature_columns: List of expected feature column names
        scaler: Fitted StandardScaler instance
        task: Type of task ('classification' or 'regression')
        categorical_columns: List of categorical column names
        
    Returns:
        Union[int, float, str]: Prediction result
        
    Raises:
        PredictionError: If prediction fails
    """
    try:
        # Validate input
        input_df = validate_input(input_data, feature_columns)
        
        # Preprocess input
        input_scaled = preprocess_input(input_df, scaler, categorical_columns)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Format prediction based on task
        if task == "classification":
            result = prediction[0]  # Return class label
        else:
            result = float(prediction[0])  # Return numeric value
            
        return result
        
    except Exception as e:
        raise PredictionError(f"Prediction failed: {str(e)}")

def get_prediction_probability(model: Any,
                             input_data: Union[Dict[str, Any], pd.DataFrame],
                             feature_columns: List[str],
                             scaler: StandardScaler,
                             categorical_columns: List[str] = None) -> Dict[str, float]:
    """
    Get prediction probabilities for classification tasks.
    
    Args:
        model: Trained scikit-learn model
        input_data: Dictionary or DataFrame containing feature values
        feature_columns: List of expected feature column names
        scaler: Fitted StandardScaler instance
        categorical_columns: List of categorical column names
        
    Returns:
        Dict[str, float]: Dictionary of class probabilities
        
    Raises:
        PredictionError: If probability calculation fails
    """
    try:
        # Validate input
        input_df = validate_input(input_data, feature_columns)
        
        # Preprocess input
        input_scaled = preprocess_input(input_df, scaler, categorical_columns)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            classes = model.classes_
            return dict(zip(classes, probabilities))
        else:
            raise PredictionError("Model does not support probability predictions")
            
    except Exception as e:
        raise PredictionError(f"Probability calculation failed: {str(e)}")

def evaluate_model(model, X_test, y_test, task):
    y_pred = model.predict(X_test)
    if task == "classification":
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1 Score": f1_score(y_test, y_pred, average='macro')
        }
    else:
        return {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2 Score": r2_score(y_test, y_pred)
        }
