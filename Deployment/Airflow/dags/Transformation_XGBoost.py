import numpy as np
import pandas as pd
import re
import ast
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def load_data(filepath):
    """Load data from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def initial_cleaning(df):
    """Perform initial data cleaning."""
    df = df.drop(['has_availability'], axis=1)
    df['review_scores_rating'] = df['review_scores_rating'].fillna(0)
    df['host_since'] = df['host_since'].fillna(df['first_review'])
    print(f"Null values in 'host_since': {df['host_since'].isnull().sum()}")
    df = df.dropna(subset=['host_since'])
    return df

def encode_categorical_columns(df):
    """Encode categorical columns into numerical codes."""
    df['property'] = df['property_type'].astype('category').cat.codes
    df['neighbourhood'] = df['neighbourhood_cleansed'].astype('category').cat.codes
    return df

def to_numerical(df):
    """Convert price column to numerical."""
    df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    return df

def extract_numerical_values(df, columns):
    """Extract numerical values from string columns."""
    for column in columns:
        df[column] = df[column].apply(
            lambda x: float(re.search(r'\d+(?:\.\d+)?', str(x)).group()) 
            if re.search(r'\d+(?:\.\d+)?', str(x)) else None
        )
    return df

def fill_missing_values(df, column_name):
    """Fill missing values with the average based on groupings."""
    average_df = df.groupby(['neighbourhood_cleansed', 'property_type', 'accommodates'])[column_name].mean().reset_index()
    average_df = average_df.rename(columns={column_name: f'average_{column_name}'})
    df = pd.merge(df, average_df, on=['neighbourhood_cleansed', 'property_type', 'accommodates'], how='left')
    df[column_name] = df[column_name].fillna(df[f'average_{column_name}'])
    df = df.drop(f'average_{column_name}', axis=1)
    return df

def process_new_host(df):
    """Process new_host column."""
    df['new_host'] = df['host_since'].str.split('-').str[0].astype(int)
    df['new_host'] = df['new_host'].apply(lambda x: 0 if x < 2024 else 1)
    df = df.drop(['host_since'], axis=1)
    return df

def map_host_response_time(df):
    """Map host response time to numerical values."""
    df['host_response_time'] = df['host_response_time'].replace(
        [np.nan, 'within an hour', 'within a few hours', 'within a day', 'a few days or more'],
        [0, 1, 2, 3, 4]
    )
    return df

def map_response_rate(rate):
    """Map response rate to numerical scores."""
    if pd.isna(rate):
        return 0
    rate = int(str(rate).strip('%'))
    if 90 <= rate <= 100:
        return 4
    elif 70 <= rate < 90:
        return 3
    elif 50 <= rate < 70:
        return 2
    elif 0 <= rate < 50:
        return 1
    else:
        return 0

def map_acceptance_rate(rate):
    """Map acceptance rate to numerical scores."""
    if pd.isna(rate):
        return 0
    rate = int(str(rate).strip('%'))
    if 90 <= rate <= 100:
        return 4
    elif 70 <= rate < 90:
        return 3
    elif 50 <= rate < 70:
        return 2
    elif 0 <= rate < 50:
        return 1
    else:
        return 0

def map_host_response_rates(df):
    """Map host response and acceptance rates."""
    df['host_response_rate'] = df['host_response_rate'].apply(map_response_rate)
    df['host_acceptance_rate'] = df['host_acceptance_rate'].apply(map_acceptance_rate)
    return df

def create_host_response_score(df):
    """Create a host response score and drop individual columns."""
    df['host_response_score'] = df['host_response_time'] + df['host_response_rate'] + df['host_acceptance_rate']
    df = df.drop(columns=['host_response_time', 'host_response_rate', 'host_acceptance_rate'])
    return df

def process_host_location(df):
    """Convert host location to binary (1 if within Canada, else 0)."""
    df['host_location'] = df['host_location'].apply(lambda x: 1 if 'Canada' in str(x) else 0)
    return df

def process_boolean_columns(df):
    """Convert 't'/'f' to 1/0 in boolean columns."""
    df['host_is_superhost'] = df['host_is_superhost'].str.contains('t', na=False).astype(int)
    df['instant_bookable'] = df['instant_bookable'].str.contains('t', na=False).astype(int)
    return df

def process_text_presence_columns(df):
    """Convert text presence columns to binary."""
    columns_to_update = [
        'host_has_profile_pic', 
        'host_identity_verified', 
        'host_about',
        'neighborhood_overview',
        'description'
    ]
    for column in columns_to_update:
        df[column] = df[column].apply(
            lambda x: 1 if x and str(x).strip() else 0
        )
    return df

def create_scores(df):
    """Create composite scores and drop individual columns."""
    df['host_score'] = (
        df['host_location'] + 
        df['host_is_superhost'] + 
        df['host_has_profile_pic'] + 
        df['host_identity_verified'] + 
        df['host_about']
    )
    df['property_description_score'] = df['neighborhood_overview'] + df['description']
    df = df.drop(columns=[
        'host_location', 'host_is_superhost', 'host_has_profile_pic', 
        'host_identity_verified', 'host_about', 
        'neighborhood_overview', 'description'
    ])
    return df

def drop_unnecessary_columns(df):
    """Drop columns that are not needed."""
    # Combine the lists of columns to drop
    columns_to_drop = [
        'neighbourhood', 'host_neighbourhood', 'host_verifications', 
        'room_type', 'source', 'bathrooms', 'first_review', 
        'last_review', 'calendar_last_scraped', 'number_of_reviews_ltm',
        'number_of_reviews_l30d', 'license', 'reviews_per_month', 
        'availability_365', 'availability_60', 'availability_90', 
        'host_id', 'listing_url', 'scrape_id', 'last_scraped', 
        'name', 'picture_url', 'host_url', 'host_name',
        'host_thumbnail_url', 'host_picture_url', 'neighbourhood_group_cleansed', 
        'calendar_updated', 'latitude', 'longitude', 'minimum_minimum_nights', 
        'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',  
        'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms', 
        'calculated_host_listings_count_shared_rooms',
        'host_listings_count', 'host_total_listings_count', 
        'review_scores_accuracy', 'review_scores_cleanliness', 
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]

    # Drop columns that exist in the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop, axis=1)
    
    return df

def add_count_amenities(df):
    """Add a column counting the number of amenities."""
    df['amenities'] = df['amenities'].fillna('[]')
    
    def parse_amenities(entry):
        try:
            amenities_list = ast.literal_eval(entry)
            return amenities_list
        except (ValueError, SyntaxError):
            return []
    
    df['amenities_list'] = df['amenities'].apply(parse_amenities)
    df['count_amenities'] = df['amenities_list'].apply(len).astype(int)
    
    return df

def process_amenities(df, top_n=100):
    """Process amenities column and select top N amenities."""
    df['amenities'] = df['amenities'].fillna('[]')

    def parse_amenities(entry):
        try:
            amenities_list = ast.literal_eval(entry)
            return amenities_list
        except (ValueError, SyntaxError):
            return []

    def normalize_amenities(amenities_list):
        normalized = []
        for amenity in amenities_list:
            amenity = amenity.lower().strip()
            amenity = amenity.replace(' ', '_').replace('"', '').replace("'", '')
            if 'translation_missing' not in amenity and amenity != '':
                normalized.append(amenity)
        return normalized

    df['amenities_list'] = df['amenities'].apply(lambda x: normalize_amenities(parse_amenities(x)))

    # Flatten the list of all amenities
    all_amenities = [amenity for sublist in df['amenities_list'] for amenity in sublist]

    # Count the occurrences of each amenity
    amenity_counts = Counter(all_amenities)

    # Calculate total number of listings
    total_listings = len(df)

    # Define frequency thresholds
    min_freq = 0.02
    max_freq = 0.98

    # Identify frequent amenities
    frequent_amenities = [
        amenity for amenity, count in amenity_counts.items()
        if min_freq <= (count / total_listings) <= max_freq
    ]

    # Filter the amenities_list to keep only frequent amenities
    df['amenities_list'] = df['amenities_list'].apply(
        lambda x: [amenity for amenity in x if amenity in frequent_amenities]
    )

    # Initialize the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit and transform the amenities_list
    amenities_encoded = mlb.fit_transform(df['amenities_list'])

    # Create a DataFrame with the encoded amenities
    amenities_df = pd.DataFrame(amenities_encoded, columns=mlb.classes_)

    # Select top N amenities based on mutual information
    selector = SelectKBest(score_func=mutual_info_regression, k=top_n)
    amenities_selected = selector.fit_transform(amenities_df, df['price'])
    selected_indices = selector.get_support(indices=True)
    selected_amenities = [mlb.classes_[i] for i in selected_indices]
    amenities_df_selected = amenities_df[selected_amenities]

    # Concatenate amenities_df_selected to df
    df = pd.concat([df.reset_index(drop=True), amenities_df_selected.reset_index(drop=True)], axis=1)

    # Drop 'amenities' and 'amenities_list' columns
    df = df.drop(columns=['amenities', 'amenities_list'])
    return df

def process_month_column(df):
    """Extract year and month from the 'month' column."""
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m-%d')
    df['data_year'] = df['month'].dt.year
    df['data_month'] = df['month'].dt.month
    df = df.drop(columns=['month'])
    return df

def drop_missing_values(df):
    """Drop rows with missing values in critical columns."""
    df = df.dropna(subset=['bedrooms', 'bathrooms_text', 'beds'])
    return df

def main_transformation(inputfilepath, outputfilepath):
    df = load_data(inputfilepath)
    df = initial_cleaning(df)
    df = encode_categorical_columns(df)
    df = to_numerical(df)
    df = fill_missing_values(df, 'price')
    df = extract_numerical_values(df, ['bathrooms_text'])
    df = fill_missing_values(df, 'bedrooms')
    df = fill_missing_values(df, 'beds')
    df = fill_missing_values(df, 'bathrooms_text')
    df = process_new_host(df)
    df = map_host_response_time(df)
    df = map_host_response_rates(df)
    df = create_host_response_score(df)
    df = process_host_location(df)
    df = process_boolean_columns(df)
    df = process_text_presence_columns(df)
    df = create_scores(df)
    df = drop_unnecessary_columns(df)
    df = process_month_column(df)
    df = drop_missing_values(df)
    df = add_count_amenities(df)
    df = process_amenities(df, top_n=100)

    # Save the processed DataFrame to CSV
    df.to_csv(f'{outputfilepath}/processed_data_xgboost.csv', index=False)
    print('done saving.')
