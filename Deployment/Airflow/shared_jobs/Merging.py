import os
from datetime import datetime
import calendar
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

def add_month_column_and_merge(base_folder, output_path, spark):
    # List all files in the base folder
    all_files = os.listdir(base_folder)
    merged_df = None

    for file_name in all_files:
        if not file_name.endswith(".csv.gz"):
            continue
        file_path = os.path.join(base_folder, file_name)
        date_str = file_name.split('_')[1].split('.')[0]
        date_obj = datetime.strptime(date_str, "%Y-%m")
        month_name = calendar.month_name[date_obj.month]
        print('Starting to read file....')

        # Read file and add 'month' columns
        df = spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            quote='"',
            escape='"'
        )
        df = df.repartition(10)
        df = df.withColumn("Month", lit(month_name))

        # Merge DataFrames
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.union(df)
        print('merged')

    # Save merged DataFrame
    merged_df.coalesce(1).write.csv(output_path, header=True)

if __name__ == "__main__":
    # Argument parsing for dynamic paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", required=True, help="Path to the folder containing monthly data files")
    parser.add_argument("--output_path", required=True, help="Path to save the merged output file")
    args = parser.parse_args()

    # Initialize Spark session
    spark = SparkSession.builder.appName("AirbnbMergePipeline").getOrCreate()

    # Merge data
    add_month_column_and_merge(args.base_folder, args.output_path, spark)

    # Stop Spark session
    spark.stop()
