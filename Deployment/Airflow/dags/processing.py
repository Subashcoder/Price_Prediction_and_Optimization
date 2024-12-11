import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import ast 
from collections import Counter

def Processing_for_model(inputfilepath, outputfilepath):
    df = pd.read_csv(inputfilepath, on_bad_lines='skip', low_memory=False)
    df['availability_30'] = pd.to_numeric(df['availability_30'], errors='coerce')
    df_final = df.drop(columns=['listing_url','last_scraped','source', 'name', 'picture_url', 'host_name', 'host_thumbnail_url','host_picture_url','calendar_last_scraped'], axis=1)
    
    bool_columns = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability', 'instant_bookable']
    for col in bool_columns:
        df_final[col] = df_final[col].map({'t': 1, 'f': 0})
        
    df_final['host_verifications'] = df_final['host_verifications'].apply(ast.literal_eval)

    all_verifications = set([item for sublist in df_final['host_verifications'] for item in sublist])

    for verification in all_verifications:
        df_final[f'has_{verification}'] = df_final['host_verifications'].apply(lambda x: 1 if verification in x else 0)


    df_final.drop(columns=['host_verifications'], inplace=True)
    df_final.drop(columns='host_url', inplace=True)
    
    import re

# Function to clean each amenities entry
    def clean_amenities(amenities_string):
        # Remove backslashes and extraneous quotes
        cleaned_string = re.sub(r'\\|"', '', amenities_string)
        # Convert string representation of list to actual Python list
        # Split by comma and strip spaces around each amenity
        amenities_list = [amenity.strip() for amenity in cleaned_string.strip("[]").split(",")]
        return amenities_list

    df_final['amenities'] = df_final['amenities'].apply(clean_amenities)
    
    Boolean_columns = ['host_is_superhost', 'host_has_profile_pic','host_identity_verified', 'has_availability','instant_bookable' ]
    df_final[Boolean_columns]
    
  
    all_amenities = [amenity.strip() for amenities_list in df_final['amenities'] for amenity in amenities_list]
    amenity_counts = Counter(all_amenities)

    # Get the top 300 amenities
    top_amenities = [amenity for amenity, count in amenity_counts.most_common(300)]
    
    def filter_top_amenities(amenities_list):
        return [amenity for amenity in amenities_list if amenity.strip() in top_amenities]

    df_final['filtered_amenities'] = df_final['amenities'].apply(filter_top_amenities)
    
    Boolean_columns = ['host_is_superhost', 'host_has_profile_pic','host_identity_verified', 'has_availability','instant_bookable' ]
    df_final[Boolean_columns]
    
    amenity_to_index = {amenity: idx for idx, amenity in enumerate(top_amenities)}

# Update multi-hot encoding function to use top amenities
    def amenities_to_multihot(amenities_list):
        multihot_vector = [0] * len(top_amenities)  # Initialize vector of length 300
        for amenity in amenities_list:
            if amenity.strip() in amenity_to_index:
                multihot_vector[amenity_to_index[amenity.strip()]] = 1
        return multihot_vector

    df_final['amenities_multihot'] = df_final['filtered_amenities'].apply(amenities_to_multihot)
    
    Boolean_columns = ['host_is_superhost', 'host_has_profile_pic','host_identity_verified', 'has_availability','instant_bookable' ]
    df_final[Boolean_columns]
    
    label_encoders = {}
    categorical_columns = ['neighbourhood_cleansed', 'room_type', 'property_type', 'Month']

    for col in categorical_columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col])
        label_encoders[col] = le
        
    df_final['ROI'] = df_final['price'] * (30 - df_final['availability_30'])

    numerical_cols = ['accommodates', 'host_response_rate', 'host_acceptance_rate', 'latitude', 'longitude', 'price', 'beds','bedrooms','bathrooms_text','host_identity_verified',
                  'minimum_nights','maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                  'number_of_reviews', 'review_scores_rating', 'reviews_per_month', 'host_is_superhost', 'host_has_profile_pic', 'has_availability','instant_bookable',
                  'calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms',
                  'calculated_host_listings_count_shared_rooms', 'ROI']

    scaler = StandardScaler()
    df_final[numerical_cols] = scaler.fit_transform(df_final[numerical_cols])
    
    tfidf = TfidfVectorizer(max_features=50)  # Adjust max_features as needed
    tfidf_matrix = tfidf.fit_transform(df_final['neighborhood_overview']).toarray()

    # Append TF-IDF columns to dataframe
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    df_final = pd.concat([df_final.reset_index(drop=True), tfidf_df], axis=1)
    
    df_final.to_csv(f'{outputfilepath}/CleanData_Afterencoding.csv')

    