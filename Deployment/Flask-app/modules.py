import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tensorflow.keras.models import load_model


# Function to retrieve embeddings
def get_embedding(model, feature, values):
    """
    Fetch embeddings for a categorical feature using the trained embedding layer.
    :param model: Trained Keras model
    :param feature: Name of the input feature (e.g., 'neighborhood')
    :param values: Categorical values (e.g., [0, 1, 2] for encoded categories)
    :return: Embedding vectors
    """
    embedding_layer = model.get_layer(f"embedding{feature}")
    
    
    embeddings = embedding_layer(np.array(values))
    return embeddings.numpy()


# Calculate user profile
def calculate_user_profile(model, user_listings, numerical_cols):
    # Retrieve embeddings for categorical features
    neighborhood_embeddings = get_embedding(model, '', user_listings['neighbourhood_cleansed'])
    room_type_embeddings = get_embedding(model, '_1', user_listings['room_type'])
    property_type_embeddings = get_embedding(model, '_2', user_listings['property_type'])
    month_embeddings = get_embedding(model, '_3', user_listings['Month'])
    cluster_embeddings = get_embedding(model, '_4', user_listings['cluster'])

    # Step 3: Aggregate embeddings (e.g., mean)
    user_profile = {
        'neighborhood': np.mean(neighborhood_embeddings, axis=0),
        'room_type': np.mean(room_type_embeddings, axis=0),
        'property_type': np.mean(property_type_embeddings, axis=0),
        'month': np.mean(month_embeddings, axis=0),
        'cluster': np.mean(cluster_embeddings, axis=0)
    }
    
    for column in numerical_cols:
        user_profile[column] = np.array([user_listings[column].mean()])
        
    user_profile_vector = np.concatenate(list(user_profile.values()))
    return user_profile_vector

# Get recommendation
def get_recommendation(model, numerical_cols, df_final, host_id, listing_vector):

    # Filteing listings for the given host_id
    user_listings = df_final[df_final['host_id'] == host_id]

    if user_listings.empty:
        raise ValueError(f"No listings found for host_id: {host_id}")

    # Calculate user profile vector
    user_profile_vector = calculate_user_profile(model, user_listings, numerical_cols)

    user_amenities_vectors = np.array(user_listings['amenities_multihot'].tolist())
    user_amenities_profile = np.mean(user_amenities_vectors, axis=0)
        

    listing_amenities_matrix = np.vstack(df_final['amenities_multihot'])

        # Compute cosine similarity
    amenities_similarities = cosine_similarity(
            [user_amenities_profile], listing_amenities_matrix
        )[0]

    feature_similarities = cosine_similarity(
            [user_profile_vector], listing_vector
        )[0]

        # Combine similarities
    df_final['feature_similarity'] = feature_similarities
    df_final['amenities_similarity'] = amenities_similarities
    df_final['combined_similarity'] = (
            0.5 * df_final['feature_similarity'] +
            0.5 * df_final['amenities_similarity']
        )

        # Calculate ranking score
    similarity_threshold = 0.6
    similar_listings = df_final[df_final['combined_similarity'] >= similarity_threshold]
    
    
    neighborhood_recommendations = similar_listings.groupby('neighbourhood_cleansed').agg({
    'combined_similarity': 'mean',
    'price': 'mean',
    'availability_30': 'mean',
    'host_id': 'count',
    'ROI': ['mean', 'std']
    
        }).reset_index()

    neighborhood_recommendations.columns = ['neighbourhood_cleansed', 'mean_similarity','mean price','Bookings','Number of similar listings','mean_ROI', 'std_ROI']

    # Add ranking score
    neighborhood_recommendations['ranking_score'] = (
        0.7 * neighborhood_recommendations['mean_similarity'] +
        0.3 * neighborhood_recommendations['mean_ROI']
        )
    
    neighborhood_recommendations = neighborhood_recommendations.sort_values(by=['ranking_score','mean_ROI','Number of similar listings'], ascending=False).head(10)
    print(neighborhood_recommendations)

    
    return neighborhood_recommendations

def prediction_neighbourhood(recommendations, df_final, host_id, model, numerical_cols):
    # Retrieve all listings for a given `host_id`
    host_id = 107788572
    user_listings = df_final[df_final['host_id'] == host_id]

    # Get all the recommended neighborhoods for the given host
    recommended_neighborhoods = recommendations['neighbourhood_cleansed'].values

    predictions_per_neighborhood = []

    for recommended_neighborhood in recommended_neighborhoods:
        # Step 3.1: Update the Neighborhood for this Recommendation
        user_listings_copy = user_listings.copy()  # Make a copy to avoid modifying the original listings
        user_listings_copy['neighbourhood_cleansed'] = recommended_neighborhood  # Replace with recommended neighborhood
        
        # Step 3.2: Prepare the features for prediction
        X = {
            'neighborhood': user_listings_copy['neighbourhood_cleansed'].values,
            'room_type': user_listings_copy['room_type'].values,
            'property_type': user_listings_copy['property_type'].values,
            'Month': user_listings_copy['Month'].values,
            'cluster': user_listings_copy['cluster'].values,
            'numerical_input': user_listings_copy[numerical_cols].values,
            'tfidf_input': user_listings[[f'tfidf_{i}' for i in range(50)]],
            'amenities_input': np.stack(user_listings_copy['amenities_multihot'].values)
        }


        # Step 3.3: Predict the ROI for this neighborhood
        predicted_roi = model.predict(X)
        
        # Step 4: Store the predictions for this neighborhood
        user_listings_copy['predicted_ROI'] = predicted_roi
        user_listings_copy['recommended_neighborhood'] = recommended_neighborhood

        # Collect all predictions for each recommended neighborhood
        predictions_per_neighborhood.append(user_listings_copy)

    # Step 5: Combine all predictions into a single DataFrame
    all_predictions = pd.concat(predictions_per_neighborhood)
    all_predictions_month = all_predictions.groupby(['recommended_neighborhood','Month']).agg({'price': 'mean', 'availability_30': 'mean', 'predicted_ROI':'mean'}).reset_index()
    all_predictions_month = all_predictions_month.sort_values('predicted_ROI', ascending=False)
    
    # No Month
    all_predictions_NoMonth = all_predictions.groupby('recommended_neighborhood').agg({'price': 'mean', 'availability_30': 'mean', 'predicted_ROI':'mean'}).reset_index()
    all_predictions_NoMonth = all_predictions_NoMonth.sort_values('predicted_ROI', ascending=False)
    # Display the final predictions DataFrame
    return  all_predictions_month, all_predictions_NoMonth
    
