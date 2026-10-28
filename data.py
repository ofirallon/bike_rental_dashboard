import pandas as pd
import json

def audit_and_prepare_data(file_path="hour.csv"):
    print("Reading bike rental data...")
    df = pd.read_csv(file_path)

    # 1. Create a Timestamp index from date and hour
    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    
    # Sort chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2. Check for duplicates
    duplicates_count = df['timestamp'].duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates(subset=['timestamp'])
    
    # 3. Find missing hours
    full_time_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='h')
    missing_hours = full_time_range.difference(df['timestamp'])
    missing_count = len(missing_hours)

    # 4. Split into periods according to instructions
    def assign_period(ts):
        if ts < pd.to_datetime("2012-05-01"):
            return "Training"
        elif ts < pd.to_datetime("2012-08-01"):
            return "Validation"
        else:
            return "Test"
            
    df['period'] = df['timestamp'].apply(assign_period)
    period_counts = df['period'].value_counts().to_dict()

    # 5. Create and save the audit report
    audit_results = {
        "total_rows_after_dedup": len(df),
        "duplicates_removed": int(duplicates_count),
        "missing_hours_count": missing_count,
        "training_hours": period_counts.get('Training', 0),
        "validation_hours": period_counts.get('Validation', 0),
        "test_hours": period_counts.get('Test', 0)
    }
    
    audit_df = pd.DataFrame([audit_results])
    audit_df.to_csv("data_audit.csv", index=False)
    print("-> Audit file 'data_audit.csv' saved successfully.")
    
    # 6. Save period boundaries to config.json
    config = {
        "periods": {
            "training": {"start": "2011-01-01", "end": "2012-04-30"},
            "validation": {"start": "2012-05-01", "end": "2012-07-31"},
            "test": {"start": "2012-08-01", "end": "2012-12-31"}
        }
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("-> Config file 'config.json' saved successfully.")

    return df

if __name__ == "__main__":
    df_cleaned = audit_and_prepare_data()
    print("Finished checking the data!")