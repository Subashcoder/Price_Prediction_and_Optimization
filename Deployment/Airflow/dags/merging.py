import os
from datetime import datetime
import calendar
import pandas as pd

def merge_csv_files(base_folder: str, output_path: str) -> None:
    """
    Merges all CSV files in the specified folder, adds a 'Month' column based on the filename,
    and saves the result to the specified output path.

    :param base_folder: Path to the folder containing input CSV files.
    :param output_path: Path where the merged CSV file will be saved.
    """
    all_files = os.listdir(base_folder)
    merged_df = pd.DataFrame()

    for file_name in all_files:
        if not file_name.endswith(".csv.gz"):
            continue
        
        file_path = os.path.join(base_folder, file_name)
        date_str = file_name.split('_')[1].split('.')[0]
        date_obj = datetime.strptime(date_str, "%Y-%m")
        month_name = calendar.month_name[date_obj.month]
        print(f"Processing file: {file_name}")

        # Read the CSV and add 'Month' column
        df = pd.read_csv(file_path, low_memory=False)
        df['Month'] = month_name

        # Merge DataFrames
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        print(f"Merged: {file_name}")

    # Save the final merged DataFrame
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")
