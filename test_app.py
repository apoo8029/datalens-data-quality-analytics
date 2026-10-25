
import pandas as pd

from app import analyze_drift


def test_identifier_is_skipped():
    reference = pd.DataFrame({
        "customer_id": [101, 102, 103],
        "age": [25, 30, 35],
    })

    current = pd.DataFrame({
        "customer_id": [201, 202, 203],
        "age": [26, 31, 36],
    })

    results = analyze_drift(reference, current)

    identifier_results = [
        result for result in results
        if result["Column"] == "customer_id"
    ]

    assert len(identifier_results) == 1
    assert identifier_results[0]["Status"] == "SKIPPED"
    assert identifier_results[0]["Metric"] == "Identifier Excluded"


def test_new_column_is_detected():
    reference = pd.DataFrame({
        "age": [20, 30, 40],
    })

    current = pd.DataFrame({
        "age": [20, 30, 40],
        "income": [100, 200, 300],
    })

    results = analyze_drift(reference, current)

    income_results = [
        result for result in results
        if result["Column"] == "income"
    ]

    assert len(income_results) == 1
    assert income_results[0]["Status"] == "NEW COLUMN"
    assert income_results[0]["Metric"] == "Schema Change"
