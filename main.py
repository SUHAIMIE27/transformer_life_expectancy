# main.py

from config import *
from src.data.data_loader import DatasetLoader, save_processed_data
from src.preprocessing.preprocessing import TransformerLifePreprocessor
from src.models.train_model import ModelTrainer, save_model
from src.evaluation.evaluate import evaluate_model, save_metrics
from src.utils.reporting import make_summary, save_text_report
from src.utils.visualization import plot_results

def main():
    # Load dataset
    loader = DatasetLoader(RAW_DATA_PATH)
    df = loader.load()
    print(f"Dataset shape: {df.shape}")

    # Preprocessing
    feature_cols = [
        'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
        'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor',
        'Interfacial V', 'Dielectric rigidity', 'Water content',
        'Health index'
    ]
    preprocessor = TransformerLifePreprocessor(feature_cols)
    processed_df = preprocessor.transform(df)
    save_processed_data(processed_df, PROCESSED_DATA_PATH)

    # Train model
    trainer = ModelTrainer(target_column=TARGET_COLUMN, random_state=RANDOM_STATE)
    results = trainer.train(processed_df)

    # Evaluate model
    metrics = evaluate_model(results["y_test"], results["preds"])
    print(f"Evaluation metrics: {metrics}")
    save_metrics(metrics, METRICS_DIR / "metrics.json")
    save_model(results["model"], MODEL_PATH)

    # Visualization
    plot_results(results["y_test"], results["preds"], output_path=FIGURE_DIR / "predicted_vs_true.png")

    # Reporting
    summary = make_summary(processed_df)
    save_text_report(summary, METRICS_DIR / "summary.txt")
    print("Pipeline Summary:")
    print(summary)

if __name__ == "__main__":
    main()