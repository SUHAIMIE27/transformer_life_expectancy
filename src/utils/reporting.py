# src/utils/reporting.py
from pathlib import Path

def make_summary(df):
    """
    Return a text summary of the dataset: shape, column info, missing values.
    """
    summary = []
    summary.append(f"Dataset shape: {df.shape}")
    summary.append("\nColumn types and non-null counts:")
    summary.append(df.info(buf=None))
    summary.append("\nMissing values per column:")
    summary.append(df.isnull().sum().to_string())
    return "\n".join([str(s) for s in summary])

def save_text_report(summary_text: str, output_path: str | Path):
    """
    Save summary text to a file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary_text)