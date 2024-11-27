# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# ---------------------------
# 1. Constants Definition
# ---------------------------

# Define price bins and labels (using underscores for consistency)
PRICE_BINS = [0, 150, 300, 500, 1000, np.inf]
PRICE_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Define columns for scaling
STANDARD_SCALER_COLUMNS = [
    'minimum_nights', 'maximum_nights', 'availability_30', 'number_of_reviews',
    'review_scores_rating', 'calculated_host_listings_count', 'count_amenities',
    'data_year', 'data_month'
]

MIN_MAX_SCALER_COLUMNS = [
    'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'host_response_score',
    'host_score', 'property_description_score'
]

# Define categorical columns (ensure all categorical columns are included)
CATEGORICAL_COLUMNS = ['neighbourhood_cleansed', 'property_type']

# Paths for saving models and scalers (current directory)
MODEL_DIR = "."  # Current directory
SCALER_DIR = "."  # Current directory
FEATURE_COLUMNS_FILE = "feature_columns.pkl"

# ---------------------------
# 2. Utility Functions
# ---------------------------

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Airbnb listings data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logging.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape {df.shape}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        st.error(f"File not found: {file_path}")
        st.stop()
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        st.stop()
    return df

@st.cache_data
def load_feature_columns(feature_columns_path: str) -> list:
    """
    Loads the feature columns from a pickle file.

    Parameters:
        feature_columns_path (str): Path to the feature columns pickle file.

    Returns:
        list: List of feature column names.
    """
    logging.info(f"Loading feature columns from {feature_columns_path}")
    try:
        feature_columns = joblib.load(feature_columns_path)
        logging.info(f"Feature columns loaded: {feature_columns}")
    except FileNotFoundError:
        logging.error(f"Feature columns file not found: {feature_columns_path}")
        st.error(f"Feature columns file not found: {feature_columns_path}")
        st.stop()
    except Exception as e:
        logging.error(f"Error loading feature columns: {e}")
        st.error(f"Error loading feature columns: {e}")
        st.stop()
    return feature_columns

def load_models_and_scalers() -> dict:
    """
    Loads all models and their corresponding scalers for each price range.

    Returns:
        dict: Dictionary containing models and scalers.
    """
    logging.info("Loading models and scalers for each price range")
    models = {}
    scalers = {}
    for label in PRICE_LABELS:
        model_path = os.path.join(MODEL_DIR, f'xgboost_model_{label}.pkl')
        standard_scaler_path = os.path.join(SCALER_DIR, f'standard_scaler_{label}.pkl')
        min_max_scaler_path = os.path.join(SCALER_DIR, f'min_max_scaler_{label}.pkl')

        # Check if all necessary files exist
        if os.path.exists(model_path) and os.path.exists(standard_scaler_path) and os.path.exists(min_max_scaler_path):
            try:
                model = joblib.load(model_path)
                standard_scaler = joblib.load(standard_scaler_path)
                min_max_scaler = joblib.load(min_max_scaler_path)
                models[label] = model
                scalers[label] = {
                    'standard_scaler': standard_scaler,
                    'min_max_scaler': min_max_scaler
                }
                logging.info(f"Loaded model and scalers for price range '{label}'")
            except Exception as e:
                logging.error(f"Error loading model/scalers for '{label}': {e}")
        else:
            logging.warning(f"Missing files for price range '{label}'. Skipping.")

    if not models:
        logging.error("No models and scalers loaded. Ensure that the .pkl files are in the current directory.")
        st.error("No models and scalers loaded. Please check that the .pkl files are present in the app directory.")
        st.stop()

    return {'models': models, 'scalers': scalers}

def preprocess_input_data(
    listing_data: pd.Series,
    availability_30_value: int,
    data_year_value: int,
    data_month_value: int,
    feature_columns: list,
    scalers: dict
) -> pd.DataFrame:
    """
    Preprocesses the input listing data for prediction.

    Parameters:
        listing_data (pd.Series): Original listing data.
        availability_30_value (int): Updated availability value.
        data_year_value (int): Updated year value.
        data_month_value (int): Updated month value.
        feature_columns (list): List of feature columns.
        scalers (dict): Dictionary containing 'standard_scaler' and 'min_max_scaler'.

    Returns:
        pd.DataFrame: Preprocessed input data ready for prediction.
    """
    # Update features
    listing_data['availability_30'] = availability_30_value
    listing_data['data_year'] = data_year_value
    listing_data['data_month'] = data_month_value


    # Convert categorical columns to string, strip spaces
    for col in CATEGORICAL_COLUMNS:
        if col in listing_data:
            listing_data[col] = str(listing_data[col]).strip()
        else:
            # Assign 'Unknown' if column is missing
            listing_data[col] = 'Unknown'

    # Drop irrelevant columns
    input_data = listing_data.drop(['id', 'Price_Range'], errors='ignore').to_frame().T

    # Handle missing columns by assigning default values
    missing_cols = set(feature_columns) - set(input_data.columns)
    for col in missing_cols:
        if col in CATEGORICAL_COLUMNS:
            input_data[col] = 'Unknown'
        else:
            input_data[col] = 0
    input_data = input_data[feature_columns]

    # Convert categorical columns to 'category' dtype
    for col in CATEGORICAL_COLUMNS:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype('category')

    # Identify and convert any remaining object dtype columns to numeric, excluding categorical columns
    object_columns = input_data.select_dtypes(include=['object']).columns.tolist()
    object_columns = [col for col in object_columns if col not in CATEGORICAL_COLUMNS]

    for col in object_columns:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0).astype(int)

    # Scale numerical features
    try:
        input_data[STANDARD_SCALER_COLUMNS] = scalers['standard_scaler'].transform(input_data[STANDARD_SCALER_COLUMNS])
    except Exception as e:
        logging.error(f"Error applying StandardScaler: {e}")
        st.error(f"Error applying StandardScaler: {e}")
        st.stop()

    try:
        input_data[MIN_MAX_SCALER_COLUMNS] = scalers['min_max_scaler'].transform(input_data[MIN_MAX_SCALER_COLUMNS])
    except Exception as e:
        logging.error(f"Error applying MinMaxScaler: {e}")
        st.error(f"Error applying MinMaxScaler: {e}")
        st.stop()

    return input_data

# ---------------------------
# 3. Prediction Class
# ---------------------------

class AirbnbPricePredictor:
    """
    A class to handle loading models and scalers, and making price predictions.
    """
    def __init__(self, models_scalers: dict, feature_columns: list):
        """
        Initializes the predictor with models, scalers, and feature columns.

        Parameters:
            models_scalers (dict): Dictionary containing models and scalers.
            feature_columns (list): List of feature columns.
        """
        self.models = models_scalers['models']
        self.scalers = models_scalers['scalers']
        self.feature_columns = feature_columns

    def predict_price(
        self,
        listing_id: int,
        availability_30_value: int,
        data_year_value: int,
        data_month_value: int,
        df: pd.DataFrame
    ) -> tuple:
        """
        Predicts the optimal price for a given listing ID with updated features.

        Parameters:
            listing_id (int): The ID of the listing.
            availability_30_value (int): Updated availability value.
            data_year_value (int): Updated year value.
            data_month_value (int): Updated month value.
            df (pd.DataFrame): DataFrame containing all listings.

        Returns:
            tuple: Predicted price and the price range label used.
        """
        logging.info(f"Starting prediction for Listing ID: {listing_id}")

        # Retrieve the listing data
        df_listing = df[df['id'] == listing_id]
        if df_listing.empty:
            logging.error(f"Listing ID {listing_id} not found.")
            st.error(f"Listing ID {listing_id} not found.")
            return None, None

        listing_data = df_listing.iloc[0].copy()

        # Determine price range based on original price
        original_price = listing_data['price']
        price_range_label = pd.cut(
            [original_price],
            bins=PRICE_BINS,
            labels=PRICE_LABELS,
            include_lowest=True
        )[0]

        st.write(f"**Original Price:** ${original_price}")
        st.write(f"**Price Range:** {price_range_label.replace('_', ' ')}")

        # Load the corresponding model and scalers
        model = self.models.get(price_range_label)
        scalers = self.scalers.get(price_range_label)

        if not model or not scalers:
            logging.error(f"No model or scalers available for price range '{price_range_label}'.")
            st.error(f"No model or scalers available for price range '{price_range_label}'.")
            return None, None

        # Preprocess input data
        input_data = preprocess_input_data(
            listing_data,
            availability_30_value,
            data_year_value,
            data_month_value,
            self.feature_columns,
            scalers
        )

        # Predict the price
        try:
            predicted_price = model.predict(input_data)[0]
            logging.info(f"Predicted Optimal Price for Listing ID {listing_id}: ${predicted_price:.2f}")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            st.error(f"Error during prediction: {e}")
            return None, None

        return predicted_price, price_range_label

# ---------------------------
# 4. Streamlit App Layout
# ---------------------------

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Airbnb Optimal Price Predictor", layout="wide")
    st.title("🏠 Airbnb Optimal Price Predictor")

    # Sidebar for user inputs
    st.sidebar.header("Input Parameters")

    # Load data
    data_file_path = "listing_1_With_Amenities.csv"  # Current directory
    df = load_data(data_file_path)

    # Create Price_Range
    df['Price_Range'] = pd.cut(df['price'], bins=PRICE_BINS, labels=PRICE_LABELS, include_lowest=True)

    # Load feature columns
    feature_columns = load_feature_columns(FEATURE_COLUMNS_FILE)

    # Load models and scalers
    models_scalers = load_models_and_scalers()

    # Initialize the predictor
    predictor = AirbnbPricePredictor(models_scalers, feature_columns)

    # User selects a Listing ID
    listing_ids = df['id'].unique().tolist()
    selected_listing_id = st.sidebar.selectbox("Select Listing ID", listing_ids)

    # Display current listing details
    selected_listing = df[df['id'] == selected_listing_id].iloc[0]
    st.subheader("📄 Current Listing Details")
    st.write(selected_listing)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Update Features")

    # User inputs for updating features
    availability_30_min = int(df['availability_30'].min())
    availability_30_max = int(df['availability_30'].max())
    availability_30_value = st.sidebar.slider(
        "Availability in Next 30 Days",
        min_value=availability_30_min,
        max_value=availability_30_max,
        value=int(selected_listing['availability_30']),
        step=1
    )

    data_year_min = int(df['data_year'].min())
    data_year_max = int(df['data_year'].max())
    data_year_value = st.sidebar.number_input(
        "Data Year",
        min_value=data_year_min,
        max_value=data_year_max,
        value=int(selected_listing['data_year']),
        step=1
    )

    data_month_value = st.sidebar.number_input(
        "Data Month",
        min_value=1,
        max_value=12,
        value=int(selected_listing['data_month']),
        step=1
    )

    st.sidebar.markdown("---")

    # Predict button
    if st.sidebar.button("Predict Optimal Price"):
        with st.spinner("Predicting..."):
            predicted_price, used_price_range = predictor.predict_price(
                listing_id=selected_listing_id,
                availability_30_value=availability_30_value,
                data_year_value=data_year_value,
                data_month_value=data_month_value,
                df=df
            )

        if predicted_price is not None:
            st.success(f"💰 **Predicted Optimal Price:** ${predicted_price:.2f}")
            st.write(f"**Used Price Range Model:** {used_price_range.replace('_', ' ')}")

            # Optional: Display Feature Importances
            if hasattr(predictor.models[used_price_range], 'feature_importances_'):
                importances = predictor.models[used_price_range].feature_importances_
                feature_importances = pd.DataFrame({
                    'Feature': predictor.feature_columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False).head(10)

                st.subheader("📊 Top 10 Feature Importances")
                st.bar_chart(feature_importances.set_index('Feature'))
        else:
            st.error("Prediction failed. Please check the input values and try again.")

    # Display prediction instructions
    st.markdown("""
    ### **How to Use This App**
    
    1. **Select a Listing ID** from the sidebar to view its current details.
    2. **Update Features** such as availability, year, and month to see how they affect the optimal price.
    3. Click on the **Predict Optimal Price** button to receive a price recommendation.
    
    **Note:** Ensure that the listing ID exists in the dataset. The prediction is based on the selected listing's current price range.
    """)

    # Optional: Display prediction instructions or model evaluation metrics
    st.markdown("---")
    st.markdown("### **About This App**")
    st.write("""
    This app predicts the optimal price for Airbnb listings based on various features. It utilizes multiple XGBoost models tailored to different price ranges to provide accurate and personalized pricing recommendations.
    """)

if __name__ == '__main__':
    main()
