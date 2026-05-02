import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
from database import get_connection


def fetch_query_logs():
    """Fetch all logged queries from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT query, confidence_score, safety_flagged, created_at
        FROM query_logs
        ORDER BY created_at
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    return rows


def build_dataframe(logs):
    """Convert logs to a DataFrame with text features for drift analysis."""
    df = pd.DataFrame(logs)
    if df.empty:
        return df

    df["query_length"] = df["query"].str.len()
    df["word_count"] = df["query"].str.split().str.len()
    df["confidence_score"] = df["confidence_score"].fillna(0.0)
    df["safety_flagged"] = df["safety_flagged"].astype(int)
    return df


def run_drift_report():
    """
    Compare two time windows of queries to detect drift.
    Reference = first half of logs, Current = second half.
    In production this would be last week vs this week.
    """
    logs = fetch_query_logs()

    if len(logs) < 10:
        print("Not enough queries logged yet to run drift analysis.")
        print(f"Current log count: {len(logs)} — need at least 10.")
        print("Run test_safety.py a few more times to generate logs, then retry.")
        return

    df = build_dataframe(logs)

    # Split into reference and current windows
    mid = len(df) // 2
    reference = df.iloc[:mid]
    current = df.iloc[mid:]

    print(f"Reference window: {len(reference)} queries")
    print(f"Current window:   {len(current)} queries")

    # Evidently drift report on numeric features
    column_mapping = ColumnMapping()
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[["query_length", "word_count", "confidence_score"]],
        current_data=current[["query_length", "word_count", "confidence_score"]],
        column_mapping=column_mapping
    )

    report.save_html("monitoring_report.html")
    print("\nDrift report saved to monitoring_report.html")
    print("Open it in your browser to see the full analysis.")

    # Print summary to console
    result = report.as_dict()
    metrics = result["metrics"]

    print("\n--- DRIFT SUMMARY ---")
    for metric in metrics:
        if "result" in metric:
            r = metric["result"]
            if "drift_by_columns" in r:
                for col, details in r["drift_by_columns"].items():
                    drifted = details.get("drift_detected", False)
                    score = details.get("drift_score", 0)
                    status = "DRIFT DETECTED" if drifted else "stable"
                    print(f"  {col}: {status} (score: {score:.4f})")


if __name__ == "__main__":
    run_drift_report()
