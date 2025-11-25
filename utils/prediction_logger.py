"""
Prediction Logger Utility
Centralized logging for production predictions
"""
import json
from datetime import datetime
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "prediction_logs.jsonl"

def save_prediction_log(customer_id, features, prediction, probability, shap_values=None):
    """
    Save a prediction to the log file
    
    Args:
        customer_id: Unique customer identifier
        features: Dictionary of customer features
        prediction: Model prediction (Yes/No or 0/1)
        probability: Churn probability (float)
        shap_values: Optional SHAP values for explainability
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "prediction": prediction,
        "probability": float(probability),
        "features": features,
        "shap_top_3": shap_values[:3].tolist() if shap_values is not None else []
    }
    
    # Create file if it doesn't exist
    if not LOG_FILE.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.touch()
    
    # Append to file
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        return True
    except Exception as e:
        print(f"Error saving prediction log: {e}")
        return False

def load_prediction_logs():
    """Load all prediction logs from JSONL file"""
    import pandas as pd
    
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            print(f"Error loading logs: {e}")
    
    return pd.DataFrame(logs) if logs else pd.DataFrame()
