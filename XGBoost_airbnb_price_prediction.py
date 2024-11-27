# airbnb_price_prediction.py

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("airbnb_price_prediction.log"),
        logging.StreamHandler()
    ]
)

# ---------------------------
# 2. Constants Definition
# ---------------------------

# Define price bins and labels
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

# Define categorical columns
CATEGORICAL_COLUMNS = ['neighbourhood_cleansed', 'property_type']

# Paths for saving models and scalers
MODEL_DIR = "models"
SCALER_DIR = "scalers"
FEATURE_COLUMNS_FILE = "feature_columns.pkl"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# ---------------------------
# 3. Utility Functions
# ---------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Airbnb listings data from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df

def create_price_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 'Price_Range' column in the DataFrame based on predefined bins and labels.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing Airbnb listings.
    
    Returns:
        pd.DataFrame: Updated DataFrame with 'Price_Range'.
    """
    logging.info("Creating 'Price_Range' column")
    df['Price_Range'] = pd.cut(df['price'], bins=PRICE_BINS, labels=PRICE_LABELS, include_lowest=True)
    logging.info("'Price_Range' column created")
    return df

def preprocess_data(
    df: pd.DataFrame,
    standard_scaler_columns: list,
    min_max_scaler_columns: list,
    categorical_columns: list
) -> tuple:
    """
    Preprocesses the DataFrame by handling missing values, scaling numerical features,
    and encoding categorical variables.
    
    Parameters:
        df (pd.DataFrame): DataFrame to preprocess.
        standard_scaler_columns (list): Columns to apply StandardScaler.
        min_max_scaler_columns (list): Columns to apply MinMaxScaler.
        categorical_columns (list): Categorical columns to convert to 'category' dtype.
    
    Returns:
        tuple: Split data and fitted scalers.
    """
    logging.info("Starting preprocessing of data")
    
    # Remove rows with missing target
    df = df.dropna(subset=['price'])
    
    # Handle 'bathrooms_text' if it's a string (convert to numeric)
    if 'bathrooms_text' in df.columns and df['bathrooms_text'].dtype == object:
        logging.info("Converting 'bathrooms_text' from object to float")
        df['bathrooms_text'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Fill missing numerical values with median
    logging.info("Filling missing numerical values with median")
    df[standard_scaler_columns] = df[standard_scaler_columns].fillna(df[standard_scaler_columns].median())
    df[min_max_scaler_columns] = df[min_max_scaler_columns].fillna(df[min_max_scaler_columns].median())
    
    # Identify object dtype columns excluding categorical columns
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    object_columns = [col for col in object_columns if col not in categorical_columns]
    
    # Convert specified object columns to numeric (assuming binary)
    for col in object_columns:
        logging.info(f"Converting '{col}' from object to int")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Initialize scalers
    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()
    
    # Apply StandardScaler
    logging.info("Applying StandardScaler to specified columns")
    df[standard_scaler_columns] = standard_scaler.fit_transform(df[standard_scaler_columns])
    
    # Apply MinMaxScaler
    logging.info("Applying MinMaxScaler to specified columns")
    df[min_max_scaler_columns] = min_max_scaler.fit_transform(df[min_max_scaler_columns])
    
    # Convert categorical columns to 'category' dtype
    logging.info("Converting categorical columns to 'category' dtype")
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    
    # Split the data into features and target
    logging.info("Splitting data into features and target")
    X = df.drop(['price', 'Price_Range', 'id'], axis=1)  # Exclude 'id' if not used
    y = df['price']
    
    # Split the data into training and testing sets
    logging.info("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logging.info("Preprocessing completed")
    return X_train, X_test, y_train, y_test, standard_scaler, min_max_scaler

def plot_actual_vs_predicted(y_test, y_pred, price_range_label):
    """
    Plots Actual vs Predicted Prices.
    
    Parameters:
        y_test (pd.Series): Actual prices.
        y_pred (np.ndarray): Predicted prices.
        price_range_label (str): Price range label.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted Prices ({price_range_label})')
    plt.savefig(f'Actual_vs_Predicted_{price_range_label}.png')
    plt.close()
    logging.info(f"Saved Actual vs Predicted plot for '{price_range_label}'")

def plot_feature_importance(model, X_train, price_range_label, top_n=10):
    """
    Plots the top N feature importances.
    
    Parameters:
        model (xgb.XGBRegressor): Trained XGBoost model.
        X_train (pd.DataFrame): Training features.
        price_range_label (str): Price range label.
        top_n (int): Number of top features to display.
    """
    importance = model.feature_importances_
    features = X_train.columns
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'][::-1], feature_importance['Importance'][::-1])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances ({price_range_label})')
    plt.savefig(f'Feature_Importance_{price_range_label}.png')
    plt.close()
    logging.info(f"Saved Feature Importance plot for '{price_range_label}'")

# ---------------------------
# 4. Model Training and Evaluation
# ---------------------------

def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    price_range_label: str
) -> tuple:
    """
    Trains the XGBoost model using RandomizedSearchCV and evaluates its performance.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        price_range_label (str): Label of the price range.
    
    Returns:
        tuple: Trained model, MAE, RMSE, MSE, R² score, and cross-validation results.
    """
    logging.info(f"Starting training for price range: '{price_range_label}'")
    
    # Define the model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        enable_categorical=True,
        eval_metric='mae',
        random_state=42
    )

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2, 3],
    }

    # Define the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Fit the random search model
    logging.info("Starting hyperparameter tuning with RandomizedSearchCV")
    random_search.fit(X_train, y_train)
    logging.info("Hyperparameter tuning completed")

    # Get the best model
    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters for '{price_range_label}': {random_search.best_params_}")

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Evaluation Metrics for '{price_range_label}': MAE={mae}, RMSE={rmse}, MSE={mse}, R2={r2}")

    # Extract and print detailed results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df['mean_absolute_error'] = -results_df['mean_test_score']
    results_df = results_df.sort_values('rank_test_score')

    logging.info(f"\nDetailed CV Results for Price Range '{price_range_label}':")
    for index, row in results_df.iterrows():
        logging.info(f"Rank: {row['rank_test_score']}")
        logging.info(f"Parameters: {row['params']}")
        logging.info(f"Mean MAE: {row['mean_absolute_error']}")
        logging.info("-" * 50)

    # Plotting
    plot_actual_vs_predicted(y_test, y_pred, price_range_label)
    plot_feature_importance(best_model, X_train, price_range_label)

    return best_model, mae, rmse, mse, r2, results_df

# ---------------------------
# 5. Prediction Class
# ---------------------------

class AirbnbPricePredictor:
    """
    A class to handle loading models and scalers, and making price predictions.
    """
    def __init__(self, feature_columns_path: str = FEATURE_COLUMNS_FILE):
        """
        Initializes the predictor by loading models and scalers.
        
        Parameters:
            feature_columns_path (str): Path to the saved feature columns file.
        """
        logging.info("Initializing AirbnbPricePredictor")
        self.feature_columns = joblib.load(feature_columns_path)
        self.models = {}
        self.scalers = {}
        for label in PRICE_LABELS:
            model_filename = os.path.join(MODEL_DIR, f'xgboost_model_{label}.pkl')
            standard_scaler_filename = os.path.join(SCALER_DIR, f'standard_scaler_{label}.pkl')
            min_max_scaler_filename = os.path.join(SCALER_DIR, f'min_max_scaler_{label}.pkl')
            try:
                self.models[label] = joblib.load(model_filename)
                self.scalers[label] = {
                    'standard_scaler': joblib.load(standard_scaler_filename),
                    'min_max_scaler': joblib.load(min_max_scaler_filename)
                }
                logging.info(f"Loaded model and scalers for '{label}'")
            except FileNotFoundError:
                logging.warning(f"Model or scalers for '{label}' not found. Skipping.")
    
    def predict(
        self,
        listing_id: int,
        availability_30_value: int,
        data_year_value: int,
        data_month_value: int,
        df: pd.DataFrame = None
    ) -> tuple:
        """
        Predicts the optimal price for a given listing ID with updated features.
        
        Parameters:
            listing_id (int): The ID of the listing.
            availability_30_value (int): Updated availability value.
            data_year_value (int): Updated year value.
            data_month_value (int): Updated month value.
            df (pd.DataFrame, optional): The DataFrame containing listings. Required if not loaded globally.
        
        Returns:
            tuple: Predicted price and the price range label used.
        """
        if df is None:
            raise ValueError("DataFrame 'df' must be provided for prediction.")
        
        # Retrieve the listing data
        df_listing = df[df['id'] == listing_id]
        if df_listing.empty:
            logging.error(f"Listing ID {listing_id} not found.")
            return None, None
    
        listing_data = df_listing.iloc[0].copy()
        listing_data['availability_30'] = availability_30_value
        listing_data['data_year'] = data_year_value
        listing_data['data_month'] = data_month_value
    
        # Determine price range based on original price
        original_price = listing_data['price']
        price_range_label = pd.cut(
            [original_price],
            bins=PRICE_BINS,
            labels=PRICE_LABELS,
            include_lowest=True
        )[0]
    
        logging.info(f"Listing ID {listing_id} falls into price range '{price_range_label}'")
    
        # Load the corresponding model and scalers
        model = self.models.get(price_range_label)
        scalers = self.scalers.get(price_range_label)
    
        if not model or not scalers:
            logging.error(f"No model or scalers available for price range '{price_range_label}'")
            return None, None
    
        # Prepare input data
        input_data = listing_data.drop(['id']).to_frame().T
    
        # Handle missing columns by assigning default values
        missing_columns = set(self.feature_columns) - set(input_data.columns)
        for col in missing_columns:
            if col in CATEGORICAL_COLUMNS:
                input_data[col] = 'Unknown'
            else:
                input_data[col] = 0
        input_data = input_data.reindex(columns=self.feature_columns, fill_value=0)
    
        # Convert categorical columns to 'category' dtype
        for col in CATEGORICAL_COLUMNS:
            input_data[col] = input_data[col].astype('category')
    
        # Identify and convert any remaining object dtype columns to numeric
        object_columns = input_data.select_dtypes(include=['object']).columns.tolist()
        for col in object_columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0).astype(int)
    
        # Scale numerical features
        input_data[STANDARD_SCALER_COLUMNS] = scalers['standard_scaler'].transform(input_data[STANDARD_SCALER_COLUMNS])
        input_data[MIN_MAX_SCALER_COLUMNS] = scalers['min_max_scaler'].transform(input_data[MIN_MAX_SCALER_COLUMNS])
    
        # Ensure all features are present and correctly ordered
        input_data = input_data[self.feature_columns]
    
        # Predict the price
        predicted_price = model.predict(input_data)[0]
    
        logging.info(f"Predicted Optimal Price for Listing ID {listing_id}: ${predicted_price:.2f}")
        return predicted_price, price_range_label

# ---------------------------
# 6. Main Execution Flow
# ---------------------------

def main():
    """
    Main function to execute the training and prediction pipeline.
    """
    # Define data file path
    data_file_path = r"C:\Users\prabh\Downloads\airbnb_project\data\listing_1_With_Amenities.csv"
    
    # Load data
    df = load_data(data_file_path)
    
    # Create Price_Range
    df = create_price_range(df)
    
    # Create DataFrames based on the price range
    df_price_ranges = {
        label: df[df['Price_Range'] == label].copy()
        for label in PRICE_LABELS
    }
    
    # Lists to collect results
    results = []
    detailed_results = []
    
    # Train and evaluate the model for each price range
    for label in PRICE_LABELS:
        logging.info(f"\nTraining model for Price Range: '{label}'")
        df_price_range = df_price_ranges[label]
        
        # Check if the DataFrame is empty
        if df_price_range.empty:
            logging.warning(f"No data available for price range '{label}'. Skipping.")
            continue
    
        # Preprocess data
        X_train, X_test, y_train, y_test, standard_scaler, min_max_scaler = preprocess_data(
            df_price_range,
            STANDARD_SCALER_COLUMNS,
            MIN_MAX_SCALER_COLUMNS,
            CATEGORICAL_COLUMNS
        )
    
        # Train and evaluate the model
        model, mae, rmse, mse, r2, cv_results = train_and_evaluate(
            X_train, X_test, y_train, y_test, label
        )
    
        # Save the results
        results.append({
            'Price_Range': label,
            'MAE': mae,
            'RMSE': rmse,
            'MSE': mse,
            'R2 Score': r2
        })
        detailed_results.append(cv_results)
    
        # Save the model and scalers
        model_filename = os.path.join(MODEL_DIR, f'xgboost_model_{label}.pkl')
        joblib.dump(model, model_filename)
        logging.info(f"Model saved to '{model_filename}'")
    
        scaler_filenames = {
            'standard_scaler': os.path.join(SCALER_DIR, f'standard_scaler_{label}.pkl'),
            'min_max_scaler': os.path.join(SCALER_DIR, f'min_max_scaler_{label}.pkl')
        }
        joblib.dump(standard_scaler, scaler_filenames['standard_scaler'])
        joblib.dump(min_max_scaler, scaler_filenames['min_max_scaler'])
        logging.info(f"Scalers saved to '{scaler_filenames['standard_scaler']}' and '{scaler_filenames['min_max_scaler']}'")
    
    # Save the feature columns used during training
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, FEATURE_COLUMNS_FILE)
    logging.info(f"Feature columns saved to '{FEATURE_COLUMNS_FILE}'")
    
    # Save evaluation results to CSV
    evaluation_results = pd.DataFrame(results)
    evaluation_results.to_csv('evaluation_results.csv', index=False)
    logging.info("Saved evaluation results to 'evaluation_results.csv'")
    
    detailed_cv_results = pd.concat(detailed_results, ignore_index=True)
    detailed_cv_results.to_csv('detailed_cv_results.csv', index=False)
    logging.info("Saved detailed cross-validation results to 'detailed_cv_results.csv'")
    
    # Initialize the predictor
    predictor = AirbnbPricePredictor()
    
    # Example prediction
    test_listing_id = df['id'].iloc[0]  # Replace with an actual listing ID
    availability_30_value = 2  # Desired availability (2, 3, or 4)
    current_year = 2024
    current_month = 11  # November
    
    predicted_price, used_price_range = predictor.predict(
        listing_id=test_listing_id,
        availability_30_value=availability_30_value,
        data_year_value=current_year,
        data_month_value=current_month,
        df=df  # Pass the DataFrame
    )
    
    if predicted_price is not None:
        logging.info(f"Predicted Optimal Price for Listing ID {test_listing_id}: ${predicted_price:.2f}")
        logging.info(f"Used model for price range: '{used_price_range}'")
    else:
        logging.error("Prediction failed.")

if __name__ == '__main__':
    main()
