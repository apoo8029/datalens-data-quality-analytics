
from pathlib import Path
import sys

import pandas as pd


def profile_dataset(file_path: str) -> None:
    path = Path(file_path)

    if not path.is_file() or path.suffix.lower() != ".csv":
        print("Error: Provide a valid CSV file path.")
        return

    try:
        data = pd.read_csv(path, nrows=100_000)

        if data.empty:
            print("The dataset is empty.")
            return

        print("\n=== DriftPulse AI: Data Quality Report ===")
        print(f"File: {path.name}")
        print(f"Rows analyzed: {len(data):,}")
        print(f"Columns: {len(data.columns):,}")
        print(f"Duplicate rows: {data.duplicated().sum():,}")

        print("\n=== Column Summary ===")
        summary = pd.DataFrame({
            "Data Type": data.dtypes.astype(str),
            "Missing Values": data.isna().sum(),
            "Missing Percent": (data.isna().mean() * 100).round(2),
            "Unique Values": data.nunique(dropna=True),
        })
        print(summary.to_string())

        print("\nProfile completed successfully.")

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Could not process the CSV file: {error}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python data_profiler.py "path\\to\\data.csv"')
    else:
        profile_dataset(sys.argv[1])
