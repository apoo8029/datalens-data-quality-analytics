
import json

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="DataLens",
    page_icon="ðŸ“Š",
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
    )
    categorical_threshold = st.number_input(
        "Categorical drift threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
    )
    st.caption("Files are analyzed locally in this app session.")


def categorical_drift(reference, current):
    reference = reference.dropna().astype(str)
    current = current.dropna().astype(str)

    if reference.empty or current.empty:
        return None

    categories = set(reference) | set(current)
    ref_dist = reference.value_counts(normalize=True)
    cur_dist = current.value_counts(normalize=True)

    return float(
        sum(
            abs(ref_dist.get(value, 0.0) - cur_dist.get(value, 0.0))
            for value in categories
        ) / 2
    )


def analyze_drift(reference_data, current_data):
    results = []

    for column in sorted(set(reference_data.columns) | set(current_data.columns)):
        if column not in reference_data.columns:
            results.append({
                "Column": column,
                "Metric": "Schema Change",
                "Drift Score": None,
                "Status": "NEW COLUMN",
            })
            continue

        if column not in current_data.columns:
            results.append({
                "Column": column,
                "Metric": "Schema Change",
                "Drift Score": None,
                "Status": "MISSING COLUMN",
            })
            continue

        if column.strip().lower() == "id" or column.strip().lower().endswith("_id"):
            results.append({
                "Column": column,
                "Metric": "Identifier Excluded",
                "Drift Score": None,
                "Status": "SKIPPED",
            })
            continue

        reference = reference_data[column]
        current = current_data[column]

        numeric = (
            pd.api.types.is_numeric_dtype(reference)
            and pd.api.types.is_numeric_dtype(current)
        )

        if numeric:
            ref_values = reference.dropna().astype(float)
            cur_values = current.dropna().astype(float)

            if ref_values.empty or cur_values.empty:
                score = None
                status = "INSUFFICIENT DATA"
                metric = "Standardized Mean Shift"
            else:
                ref_std = float(ref_values.std(ddof=0))
                difference = abs(float(cur_values.mean()) - float(ref_values.mean()))

                if ref_std == 0:
                    score = 0.0 if difference == 0 else float("inf")
                else:
                    score = difference / ref_std

                status = "WARNING" if score > numeric_threshold else "PASS"
                metric = "Standardized Mean Shift"
        else:
            score = categorical_drift(reference, current)
            status = (
                "INSUFFICIENT DATA" if score is None
                else "WARNING" if score > categorical_threshold
                else "PASS"
            )
            metric = "Categorical Distribution Change"

        results.append({
            "Column": column,
            "Metric": metric,
            "Drift Score": score,
            "Status": status,
        })

    return results


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
    reference_data = pd.read_csv(reference_file, nrows=100_000)
    current_data = pd.read_csv(current_file, nrows=100_000)

    if reference_data.empty or current_data.empty:
        st.error("Both datasets must contain at least one data row.")
        st.stop()

    st.subheader("Dataset Comparison")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Reference Rows", f"{len(reference_data):,}")
    metric_columns[1].metric("Current Rows", f"{len(current_data):,}")
    metric_columns[2].metric("Reference Columns", len(reference_data.columns))
    metric_columns[3].metric("Current Columns", len(current_data.columns))

    missing_cells = int(current_data.isna().sum().sum())
    duplicate_rows = int(current_data.duplicated().sum())

    invalid_age_count = 0
    if "age" in current_data.columns:
        ages = pd.to_numeric(current_data["age"], errors="coerce")
        invalid_age_count = int(
            (ages.notna() & ~ages.between(0, 120)).sum()
        )

    invalid_numeric_counts = {}
    for column in current_data.columns:
        if pd.api.types.is_numeric_dtype(current_data[column]):
            values = current_data[column].dropna()
            invalid_numeric_counts[column] = int(
                (~np.isfinite(values.astype(float))).sum()
            )

    st.subheader("Current Dataset Quality")
    quality_columns = st.columns(3)
    quality_columns[0].metric("Missing Cells", missing_cells)
    quality_columns[1].metric("Duplicate Rows", duplicate_rows)
    quality_columns[2].metric("Invalid Age Values", invalid_age_count)

    st.subheader("Dataset Preview")
    st.dataframe(current_data.head(100), use_container_width=True, hide_index=True)

    quality_report = pd.DataFrame({
        "Data Type": current_data.dtypes.astype(str),
        "Missing Values": current_data.isna().sum(),
        "Missing Percent": (current_data.isna().mean() * 100).round(2),
        "Unique Values": current_data.nunique(dropna=True),
    })
    st.subheader("Column Quality Report")
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

    status_columns = st.columns(3)
    status_columns[0].metric("Potential Drift Issues", warning_count)
    status_columns[1].metric("Insufficient Data", insufficient_count)
    status_columns[2].metric("Identifier Columns Skipped", int(
        (drift_table["Status"] == "SKIPPED").sum()
    ))

    if warning_count:
        st.warning(f"{warning_count} potential drift or schema issue(s) detected.")
    else:
        st.success("No configured drift or schema warnings detected.")

    report_results = []
    for item in drift_results:
        entry = item.copy()
        score = entry["Drift Score"]
        if score is not None and not np.isfinite(score):
            entry["Drift Score"] = "Infinity"
        report_results.append(entry)

    report = {
        "project": "DataLens",
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
        },
        "thresholds": {
            "numeric_drift": numeric_threshold,
            "categorical_drift": categorical_threshold,
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
        "Numeric drift uses standardized mean shift. Categorical drift uses "
        "total variation distance. Thresholds are heuristics, not formal "
        "statistical tests. Each CSV is limited to 100,000 rows."
    )

except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Unable to analyze the uploaded datasets: {error}")

