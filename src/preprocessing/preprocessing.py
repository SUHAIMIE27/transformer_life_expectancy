# src/preprocessing/preprocessing.py
from sklearn.preprocessing import StandardScaler
import pandas as pd

class TransformerLifePreprocessor:
    """
    Preprocess transformer life-expectancy Kaggle dataset.
    Scales numeric features using StandardScaler.
    """

    def __init__(self, feature_cols=None):
        # If no feature columns specified, use all except target
        if feature_cols is None:
            self.feature_cols = [
                'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
                'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor',
                'Interfacial V', 'Dielectric rigidity', 'Water content',
                'Health index'
            ]
        else:
            self.feature_cols = feature_cols

        self.scaler = StandardScaler()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric feature columns and return processed DataFrame.
        """
        processed = data.copy()

        # Make sure all feature columns exist
        missing_cols = set(self.feature_cols) - set(processed.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        # Scale features
        processed[self.feature_cols] = self.scaler.fit_transform(processed[self.feature_cols])
        return processed