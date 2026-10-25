# DataLens — Data Quality & Drift Analytics



A Streamlit-based data quality and dataset drift monitoring dashboard.



\## Features



\- Upload and compare two CSV datasets

\- Compare dataset row and column counts

\- Detect missing values and duplicate rows

\- Identify invalid age values

\- Analyze numerical distribution changes

\- Detect categorical distribution changes

\- Identify new and missing columns

\- Exclude identifier columns from drift analysis

\- Configure drift thresholds

\- Export analysis results as a JSON report

\- Process uploaded datasets locally within the app session



\## Technology Stack



\- Python

\- Pandas

\- NumPy

\- Streamlit



\## Installation



1\. Clone this repository.

2\. Create and activate a Python virtual environment.

3\. Install dependencies:



&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. Start the application:



&#x20;  ```bash

&#x20;  streamlit run app.py

&#x20;  ```



5\. Upload a reference CSV and a current CSV to begin analysis.



\## Sample Data



Use `sample\_data.csv` as the reference dataset and `current\_data.csv` as the current dataset.



\## Drift Detection



\- Numerical columns use standardized mean shift.

\- Categorical columns use total variation distance.

\- Configurable thresholds determine potential drift warnings.



These thresholds are heuristic indicators, not formal statistical tests.



\## Limitations



\- CSV uploads are limited to 100,000 rows per dataset.

\- Data is analyzed within the application session.

\- Data quality checks depend on the available columns and their data types.



\## Future Improvements



\- Add automated unit tests

\- Add interactive drift visualizations

\- Support downloadable CSV quality reports

\- Add configurable rules for additional data quality checks



\## License



This project is available for educational and portfolio use.


