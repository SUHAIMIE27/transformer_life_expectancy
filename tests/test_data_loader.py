# tests/test_data_loader.py
import pytest
from pathlib import Path
from src.data.data_loader import DatasetLoader

def test_load():
    # Use Path to the Kaggle dataset
    file_path = Path("data/raw/kaggle_transformer_dataset.csv")
    
    loader = DatasetLoader(file_path)
    df = loader.load()
    
    # Test that the dataframe is not empty
    assert not df.empty, "Loaded dataframe is empty"
    
    # Test that expected columns exist in the dataset
    expected_columns = [
        'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene',
        'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V',
        'Dielectric rigidity', 'Water content', 'Health index', 'Life expectation'
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"