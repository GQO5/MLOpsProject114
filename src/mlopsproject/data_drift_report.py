import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.test_preset import DataDriftTestPreset

# 1. Load Dummy Data (or your real CSV)
# (Same data loading logic as before...)
reference_data = pd.DataFrame({
    'total_calories': [200, 250, 300, 220, 280],
    'total_fat': [10, 12, 15, 11, 14],
    'total_carb': [20, 25, 30, 22, 28],
    'total_protein': [5, 6, 8, 5, 7]
})

try:
    current_data = pd.read_csv("collected_data/prediction_database.csv")
    prediction_cols = ['total_calories', 'total_fat', 'total_carb', 'total_protein']
    current_data = current_data[prediction_cols]
    reference_data = reference_data[prediction_cols]
except FileNotFoundError:
    print("Warning: Using dummy data for demonstration (CSV not found).")
    current_data = reference_data.copy() # Just to make it run

# --- METHOD 1: The Visual Report (For Humans/Homework) ---
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("reports/data_drift_report.html")
print("[v] Visual Report saved to reports/data_drift_report.html")


# --- METHOD 2: The Test Suite (For Automation/GitHub Actions) ---
tests = TestSuite(tests=[
    DataDriftTestPreset()  # Returns Pass/Fail based on statistical tests
])
tests.run(reference_data=reference_data, current_data=current_data)

# Print a simple summary for the logs
print("\n--- Automation Test Results ---")
if not tests.as_dict()["summary"]["all_passed"]:
    print("[!] FAIL: Data Drift Detected!")
    # exit(1)  # Uncomment this to block the pipeline if drift is found
else:
    print("[ok] PASS: No significant drift detected.")
