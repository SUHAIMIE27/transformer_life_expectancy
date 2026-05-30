# src/evaluation/evaluate.py
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Evaluate regression model using RMSE and R2.
    """
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # R2
    r2 = r2_score(y_true, y_pred)
    
    metrics = {"RMSE": rmse, "R2": r2}
    return metrics

def save_metrics(metrics: dict, output_path: str | Path):
    """
    Save evaluation metrics to JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)