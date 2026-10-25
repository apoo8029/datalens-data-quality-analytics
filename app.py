import json

import numpy as np
import pandas as pd
import streamlit as st


MAX_ROWS = 100_000
IDENTIFIER_NAMES = {"id"}


st.set_page_config(
    page_title="DataLens",
    page_icon="📊",
    layout="wide",
)

st.title("DataLens")
st.caption("Data Quality, Dataset Comparison and Drift Monitoring")
st.write(
    "Compare datasets, identify quality issues, detect distribution "
    "changes, and export an analysis report."
)

with st.sidebar:
    st.header("Analysis Settings")
    numeric_threshold = st.number_input(
        "Numeric drift threshold",
        min_value=0.0,
        value=0.2,
        step=0.1,
        help="Warning threshold for absolute mean shift divided by reference standard deviation.",
    )
    categorical_threshold = st.number_input(
        "Categorical drift threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Warning threshold for total variation distance between category distributions.",
    )
    st.caption(f"At most {MAX_ROWS:,} rows per CSV are analyzed in this session.")


def is_identifier_column(column_name):
    """Return whether a column name looks like a record identifier."""
    normalized_name = str(column_name).strip().lower()
    return normalized_name in IDENTIFIER_NAMES or normalized_name.endswith("_id")


def categorical_drift(reference, current):
    """Calculate total variation distance between two categorical distributions."""
    reference = reference.dropna().astype(str)
    current = current.dropna().astype(str)

    if reference.empty or current.empty:
        return None

    categories = set(reference.unique()) | set(current.unique())
    reference_distribution = reference.value_counts(normalize=True)
    current_distribution = current.value_counts(normalize=True)

    return float(
        sum(
            abs(
                float(reference_distribution.get(category, 0.0))
                - float(current_distribution.get(category, 0.0))
            )
            for category in categories
        )
        / 2
    )


def numeric_drift(reference, current):
    """Calculate absolute standardized mean shift using reference standard deviation."""
    reference_values = pd.to_numeric(reference, errors="coerce")
    current_values = pd.to_numeric(current, errors="coerce")

    reference_values = reference_values.replace([np.inf, -np.inf], np.nan).dropna()
    current_values = current_values.replace([np.inf, -np.inf], np.nan).dropna()

    if reference_values.empty or current_values.empty:
        return None

    reference_mean = float(reference_values.mean())
    current_mean = float(current_values.mean())
    reference_std = float(reference_values.std(ddof=0))
    mean_difference = abs(current_mean - reference_mean)

    if reference_std == 0:
        return 0.0 if mean_difference == 0 else None

    score = mean_difference / reference_std
    return float(score) if np.isfinite(score) else None


def analyze_drift(reference_data, current_data):
    """Compare shared columns and report drift and schema changes."""
    results = []

    all_columns = sorted(
        set(reference_data.columns) | set(current_data.columns),
        key=lambda value: str(value).lower(),
    )

    for column in all_columns:
        if column not in reference_data.columns:
            results.append(
                {
                    "Column": column,
                    "Metric": "Schema Change",
                    "Drift Score": None,
                    "Status": "NEW COLUMN",
                }
            )
            continue

        if column not in current_data.columns:
            results.append(
                {
                    "Column": column,
                    "Metric": "Schema Change",
                    "Drift Score": None,
                    "Status": "MISSING COLUMN",
                }
            )
            continue

        if is_identifier_column(column):
            results.append(
                {
                    "Column": column,
                    "Metric": "Identifier Excluded",
                    "Drift Score": None,
                    "Status": "SKIPPED",
                }
            )
            continue

        reference = reference_data[column]
        current = current_data[column]
        both_numeric = (
            pd.api.types.is_numeric_dtype(reference)
            and pd.api.types.is_numeric_dtype(current)
        )

        if both_numeric:
            score = numeric_drift(reference, current)
            metric = "Standardized Mean Shift"
            if score is None:
                status = "INSUFFICIENT DATA"
            else:
                status = "WARNING" if score > numeric_threshold else "PASS"
        else:
            score = categorical_drift(reference, current)
            metric = "Categorical Distribution Change"
            if score is None:
                status = "INSUFFICIENT DATA"
            else:
                status = "WARNING" if score > categorical_threshold else "PASS"

        results.append(
            {
                "Column": column,
                "Metric": metric,
                "Drift Score": score,
                "Status": status,
            }
        )

    return results


def get_quality_report(data):
    """Create a per-column quality summary."""
    return pd.DataFrame(
        {
            "Data Type": data.dtypes.astype(str),
            "Missing Values": data.isna().sum(),
            "Missing Percent": (data.isna().mean() * 100).round(2),
            "Unique Values": data.nunique(dropna=True),
        }
    )


def count_invalid_numeric_values(data):
    """Count positive or negative infinity in numeric columns."""
    counts = {}
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            values = pd.to_numeric(data[column], errors="coerce")
            counts[column] = int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum())
    return counts


reference_file = st.file_uploader(
    "Upload Reference Dataset (CSV)",
    type=["csv"],
    key="reference",
)
current_file = st.file_uploader(
    "Upload Current Dataset (CSV)",
    type=["csv"],
    key="current",
)

if reference_file is None or current_file is None:
    st.info("Upload both CSV files to begin the analysis.")
    st.stop()

try:
    reference_data = pd.read_csv(reference_file, nrows=MAX_ROWS)
    current_data = pd.read_csv(current_file, nrows=MAX_ROWS)

    if reference_data.empty or current_data.empty:
        st.error("Both datasets must contain at least one data row.")
        st.stop()

    if reference_data.columns.duplicated().any() or current_data.columns.duplicated().any():
        st.error("Duplicate column names are not supported. Rename duplicate columns and upload again.")
        st.stop()

    st.subheader("Dataset Comparison")
    comparison_columns = st.columns(4)
    comparison_columns[0].metric("Reference Rows", f"{len(reference_data):,}")
    comparison_columns[1].metric("Current Rows", f"{len(current_data):,}")
    comparison_columns[2].metric("Reference Columns", len(reference_data.columns))
    comparison_columns[3].metric("Current Columns", len(current_data.columns))

    missing_cells = int(current_data.isna().sum().sum())
    duplicate_rows = int(current_data.duplicated().sum())

    invalid_age_count = 0
    if "age" in current_data.columns.str.lower().tolist():
        age_column = next(
            column for column in current_data.columns
            if str(column).strip().lower() == "age"
        )
        ages = pd.to_numeric(current_data[age_column], errors="coerce")
        invalid_age_count = int(
            (ages.notna() & ~ages.between(0, 120)).sum()
        )

    invalid_numeric_counts = count_invalid_numeric_values(current_data)
    total_invalid_numeric = sum(invalid_numeric_counts.values())

    st.subheader("Current Dataset Quality")
    quality_columns = st.columns(4)
    quality_columns[0].metric("Missing Cells", f"{missing_cells:,}")
    quality_columns[1].metric("Duplicate Rows", f"{duplicate_rows:,}")
    quality_columns[2].metric("Invalid Age Values", f"{invalid_age_count:,}")
    quality_columns[3].metric("Infinite Numeric Values", f"{total_invalid_numeric:,}")

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(
            current_data.head(100),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Column Quality Report")
    quality_report = get_quality_report(current_data)
    st.dataframe(quality_report, use_container_width=True)

    st.subheader("Data Drift Results")
    drift_results = analyze_drift(reference_data, current_data)
    drift_table = pd.DataFrame(drift_results)
    st.dataframe(drift_table, use_container_width=True, hide_index=True)

    warning_statuses = {"WARNING", "NEW COLUMN", "MISSING COLUMN"}
    warning_count = int(drift_table["Status"].isin(warning_statuses).sum())
    insufficient_count = int(
        (drift_table["Status"] == "INSUFFICIENT DATA").sum()
    )
    skipped_count = int((drift_table["Status"] == "SKIPPED").sum())
    pass_count = int((drift_table["Status"] == "PASS").sum())

    status_columns = st.columns(4)
    status_columns[0].metric("Potential Drift / Schema Issues", warning_count)
    status_columns[1].metric("Passed Checks", pass_count)
    status_columns[2].metric("Insufficient Data", insufficient_count)
    status_columns[3].metric("Identifier Columns Skipped", skipped_count)

    if warning_count:
        st.warning(f"{warning_count} potential drift or schema issue(s) detected.")
    else:
        st.success("No configured drift or schema warnings detected.")

    if invalid_numeric_counts and total_invalid_numeric:
        st.warning("Infinite values were found in numeric columns.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Column": column, "Infinite Values": count}
                    for column, count in invalid_numeric_counts.items()
                    if count > 0
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    report_results = []
    for item in drift_results:
        entry = item.copy()
        score = entry["Drift Score"]
        if score is not None and not np.isfinite(score):
            entry["Drift Score"] = None
        report_results.append(entry)

    report = {
        "project": "DataLens",
        "report_version": 1,
        "reference_file": reference_file.name,
        "current_file": current_file.name,
        "rows_analyzed": {
            "reference": len(reference_data),
            "current": len(current_data),
        },
        "columns": {
            "reference": list(reference_data.columns),
            "current": list(current_data.columns),
        },
        "quality": {
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "invalid_age_count": invalid_age_count,
            "infinite_numeric_values": total_invalid_numeric,
            "infinite_numeric_values_by_column": invalid_numeric_counts,
            "column_quality": quality_report.reset_index(names="Column").to_dict(orient="records"),
        },
        "thresholds": {
            "numeric_drift": numeric_threshold,
            "categorical_drift": categorical_threshold,
        },
        "summary": {
            "potential_drift_or_schema_issues": warning_count,
            "passed_checks": pass_count,
            "insufficient_data": insufficient_count,
            "identifier_columns_skipped": skipped_count,
        },
        "drift_results": report_results,
    }

    st.download_button(
        "Download Full Report (JSON)",
        data=json.dumps(report, indent=4, allow_nan=False),
        file_name="datalens_report.json",
        mime="application/json",
    )

    st.caption(
        "Numeric drift uses absolute standardized mean shift based on the reference "
        "standard deviation. Categorical drift uses total variation distance. "
        "Thresholds are heuristics, not formal statistical tests. Numeric columns "
        "with zero reference variance and a changed mean are reported as insufficient "
        "data for this metric. Each CSV is limited to 100,000 rows."
    )

except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Unable to analyze the uploaded datasets: {error}")
