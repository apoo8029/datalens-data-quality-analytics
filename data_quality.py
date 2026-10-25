
import json
import sys
from pathlib import Path

import pandas as pd


def run_quality_checks(file_path: str) -> None:
    path = Path(file_path)

    if not path.is_file() or path.suffix.lower() != ".csv":
        print("Error: Provide a valid CSV file path.")
        return

    try:
        data = pd.read_csv(path, nrows=100_000)

        if data.empty:
            print("The dataset is empty.")
            return

        checks = []

        missing_cells = int(data.isna().sum().sum())
        duplicate_rows = int(data.duplicated().sum())

        checks.append({
            "check": "Missing Values",
            "count": missing_cells,
            "status": "PASS" if missing_cells == 0 else "WARNING",
        })

        checks.append({
            "check": "Duplicate Rows",
            "count": duplicate_rows,
            "status": "PASS" if duplicate_rows == 0 else "WARNING",
        })

        if "age" in data.columns:
            age = pd.to_numeric(data["age"], errors="coerce")
            invalid_age = int(((age < 0) | (age > 120)).sum())

            checks.append({
                "check": "Invalid Ages (outside 0-120)",
                "count": invalid_age,
                "status": "PASS" if invalid_age == 0 else "WARNING",
            })

        if "customer_id" in data.columns:
            missing_ids = int(data["customer_id"].isna().sum())
            duplicate_ids = int(
                data["customer_id"].dropna().duplicated().sum()
            )

            checks.append({
                "check": "Missing Customer IDs",
                "count": missing_ids,
                "status": "PASS" if missing_ids == 0 else "WARNING",
            })

            checks.append({
                "check": "Duplicate Customer IDs",
                "count": duplicate_ids,
                "status": "PASS" if duplicate_ids == 0 else "WARNING",
            })

        print("\n=== DriftPulse AI: Quality Engine ===")
        print(f"File: {path.name}")
        print(f"Rows analyzed: {len(data):,}")
        print(f"Columns: {len(data.columns):,}")

        print("\n=== Quality Checks ===")
        for check in checks:
            print(
                f"[{check['status']}] {check['check']}: "
                f"{check['count']}"
            )

        report_path = Path("quality_report.json")
        report = {
            "file": path.name,
            "rows_analyzed": len(data),
            "columns": len(data.columns),
            "checks": checks,
        }
        report_path.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        print(f"\nReport saved to: {report_path.resolve()}")

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Could not process the CSV file: {error}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python data_quality.py "path\\to\\data.csv"')
    else:
        run_quality_checks(sys.argv[1])
