from pathlib import Path

# ===============================
# Paths
# ===============================

# Root directory for the project
PROJECT_DIR = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "kaggle_transformer_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_life_expectancy.csv"

# Outputs directories
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_PATH = OUTPUT_DIR / "model" / "life_expectancy_model.joblib"

# ===============================
# ML Pipeline parameters
# ===============================

# Features to use for preprocessing
TEXT_COLUMNS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor',
    'Interfacial V', 'Dielectric rigidity', 'Water content', 'Health index'
]

# Target column
TARGET_COLUMN = "Life expectation"

# Random state for reproducibility
RANDOM_STATE = 42

# Test split
TEST_SIZE = 0.2