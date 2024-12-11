import os
import pandas as pd
import numpy as np
from datetime import datetime


def transform_data(file_path: str, output_path: str) -> None:
    # Read CSV file into Pandas DataFrame
    print("Reading data...")
    df = pd.read_csv(file_path, low_memory=False)
    print("Data read successfully.")

    # Handling symbols and converting price to float
    print("Handling symbols and converting price to float...")
    df['price'] = df['price'].str.replace(r'[\$,]', '', regex=True).astype(float)
    print("Done.")

    # Removing text from numerical columns
    print("Removing text from 'bathrooms_text'...")
    df['bathrooms_text'] = df['bathrooms_text'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)
    print("Done.")

    # Removing percentage signs and converting to float
    print("Converting percentage columns to float...")
    percentage_columns = ['host_response_rate', 'host_acceptance_rate']
    for col in percentage_columns:
        # Ensure that only valid values remain, and replace problematic ones with NaN
        df[col] = (
            df[col]
            .str.replace('%', '', regex=False)  # Remove the '%' sign
            .replace('', np.nan)               # Replace empty strings with NaN
            .astype(float)                     # Convert to float
        )
    print("Done.")

    # Trimming whitespace
    print("Trimming whitespace from all columns...")
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    print("Done.")

    # Converting numerical columns to float
    numerical_columns = [
        'id', 'scrape_id', 'host_id', 'latitude', 'longitude', 'accommodates', 'bedrooms',
        'beds', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
        'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
        'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30',
        'availability_60', 'availability_90', 'availability_365', 'number_of_reviews',
        'number_of_reviews_ltm', 'number_of_reviews_l30d', 'calculated_host_listings_count',
        'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms', 'reviews_per_month'
    ]
    print("Converting numerical columns to float...")
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Done.")

    # Filling null values with the average price
    print("Filling null prices with average price...")
    average_prices = df.groupby(['neighbourhood_cleansed', 'property_type', 'accommodates'])['price'].transform('mean')
    df['price'] = df['price'].fillna(average_prices)
    df.dropna(subset=['price'], inplace=True)
    print("Done.")

    # Replace empty strings with NaN
    print("Replacing empty strings with NaN...")
    df.replace('', np.nan, inplace=True)

    # Fill 'beds' and 'bedrooms' with group averages
    print("Filling 'beds' and 'bedrooms' null values...")
    averages = df.groupby(['room_type', 'accommodates'])[['beds', 'bedrooms']].transform('mean')
    df[['beds', 'bedrooms']] = df[['beds', 'bedrooms']].fillna(averages)
    print("Done.")

    print("Filling response columns...")
    response_columns = ['host_response_time', 'host_response_rate', 'host_acceptance_rate']
    for col in response_columns:
        if col in df.columns:
        # Convert column to numeric, coercing invalid values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with the column mean (only for numeric values)
            df[col].fillna(df[col].mean(), inplace=True)
    print("Done.")

    # Dropping unwanted columns
    print("Dropping unwanted columns...")
    unwanted_columns = ['description', 'host_location', 'host_about', 'neighbourhood',
                        'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'host_neighbourhood']
    df.drop(columns=unwanted_columns, inplace=True, errors='ignore')
    print("Done.")

    # Filling null values for specific columns
    print("Filling specific null values...")
    df['license'] = df['license'].fillna('0')
    df['host_response_time'] = df['host_response_time'].fillna(5)
    print("Done.")

    # Handling 'neighborhood_overview'
    print("Filling 'neighborhood_overview' column...")
    neighborhood_overview = df.groupby("neighbourhood_cleansed")['neighborhood_overview'].transform('first')
    df['neighborhood_overview'] = df['neighborhood_overview'].fillna(neighborhood_overview).fillna("UNKNOWN")
    print("Done.")

    # Creating 'has_availability' column
    print("Creating 'has_availability' column...")
    df['has_availability'] = np.where(
        (df['availability_30'] == 0) & 
        (df['availability_60'] == 0) & 
        (df['availability_90'] == 0) & 
        (df['availability_365'] == 0), 'f', 't'
    )
    print("Done.")

    # Fill review columns with 0
    print("Filling review columns...")
    review_columns = [
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication", "review_scores_location",
        "review_scores_value", "reviews_per_month"
    ]
    df[review_columns] = df[review_columns].fillna(0.0)
    print("Done.")

    # Drop columns with >15% null values
    print("Dropping columns with >15% null values...")
    null_percentages = df.isnull().mean()
    columns_to_keep = null_percentages[null_percentages <= 0.15].index.tolist()
    df = df[columns_to_keep]
    print("Done.")

    # Drop remaining rows with null values
    print("Dropping remaining null rows...")
    df.dropna(inplace=True)
    print("Done.")

    # Save cleaned data
    print("Saving cleaned data...")
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
