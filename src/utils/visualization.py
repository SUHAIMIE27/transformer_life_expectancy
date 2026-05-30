# src/utils/visualization.py
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(y_true, y_pred, output_path: str | Path):
    """
    Scatter plot of true vs predicted values for regression.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Predicted vs True Values")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()