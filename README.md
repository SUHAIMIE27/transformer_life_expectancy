# Transformer Life Expectancy Prediction

This project implements a **machine learning pipeline** to predict life expectancy using a Kaggle transformer dataset. The pipeline includes **data loading, preprocessing, model training, evaluation, visualization, and reporting**.

---

## Project Structure
transformer_life_expectancy/
├── main.py # Entry point to run full pipeline
├── config.py # Dataset paths, output dirs, and hyperparameters
├── requirements.txt # Python dependencies: pandas, sklearn, matplotlib, seaborn, joblib, pytest
├── README.md

├── data/
│ ├── raw/kaggle_transformer_dataset.csv # Kaggle dataset
│ └── processed/ # Preprocessed dataset output

├── src/
│ ├── data/
│ │ └── data_loader.py # DatasetLoader & save_processed_data
│ ├── preprocessing/
│ │ └── preprocessing.py # TransformerLifePreprocessor
│ ├── models/
│ │ └── train_model.py # ModelTrainer & save_model
│ ├── evaluation/
│ │ └── evaluate.py # evaluate_model & save_metrics
│ └── utils/
│ ├── reporting.py # make_summary & save_text_report
│ └── visualization.py # plot_results

├── tests/
│ └── test_data_loader.py # Basic test for DatasetLoader

└── outputs/
├── figures/ # Generated plots
├── metrics/ # JSON and summary text
└── model/ # Saved model file


---

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/SUHAIMIE27/transformer_life_expectancy.git
cd transformer_life_expectancy
Create and activate a virtual environment:
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate
Install dependencies:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Ensure dataset is in place:
data/raw/kaggle_transformer_dataset.csv
Running the Pipeline

Run the full machine learning pipeline:

python main.py

Expected output:

Dataset shape: (470, 16)
Evaluation metrics: {'RMSE': ..., 'R2': ...}
Pipeline Summary:
...
Running Tests

To ensure the dataset loader works correctly:

python -m pytest tests/ -q

Expected output:

1 passed in 0.35s
Output

After running the pipeline, you will have:

Preprocessed dataset in data/processed/
Trained model in outputs/model/
Evaluation metrics in outputs/metrics/
Plots in outputs/figures/
Notes
Preprocessing uses StandardScaler on all numeric features.
Model uses a basic regression pipeline.
Evaluation metrics include RMSE and R².
Visualizations include predicted vs true life expectancy plots.

✅ Next Steps
Ensure the Kaggle dataset is in data/raw/kaggle_transformer_dataset.csv.
All Python modules (DatasetLoader, TransformerLifePreprocessor, ModelTrainer, evaluate_model, plot_results, make_summary) should contain the latest code.
Commit and push all changes to your GitHub repository.
Test by running:
python main.py
python -m pytest tests/ -q

Everything should work in a fresh Codespace without manual installation.



---
