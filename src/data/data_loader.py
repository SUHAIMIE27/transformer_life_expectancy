from pathlib import Path
import pandas as pd

class DatasetLoader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found")
        return pd.read_csv(self.file_path)

def save_processed_data(data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)