from flask import Flask, render_template, request, redirect
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from modules import get_recommendation, prediction_neighbourhood
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px
import ast
from tensorflow.keras.models import load_model


model = load_model("./Neural_Network/model_with_embeddings.keras")
print("Model loaded successfully!")

# Load precomputed embeddings
listing_vector = np.load('./Neural_Network/listing_vectors.npy')

# Numerical columns and categorical columns
numerical_cols = ['accommodates', 'host_response_rate', 'host_acceptance_rate', 'latitude', 'longitude', 'price', 'beds','bedrooms','bathrooms_text','host_identity_verified',
                  'minimum_nights','maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365',
                  'number_of_reviews', 'review_scores_rating', 'reviews_per_month', 'host_is_superhost', 'host_has_profile_pic', 'has_availability','instant_bookable',
                  'calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms',
                  'calculated_host_listings_count_shared_rooms', 'ROI']
categorical_columns = ['neighbourhood_cleansed', 'room_type', 'property_type', 'Month']

with open('./Neural_Network/transformations.pkl', 'rb') as file:
    transformations = pickle.load(file)

neighborhood_encoder = transformations['label_encoders']['neighbourhood_cleansed']
month_encoder = transformations['label_encoders']['Month']

df_final = pd.read_pickle('processed_data.pkl')
print("Loading done")

def plotly_graph(df):  
    scaler = transformations['scaler']
    df[numerical_cols] = scaler.inverse_transform(df[numerical_cols])  
    df_grouped = df.groupby(['neighbourhood_cleansed','Month']).agg({'price': 'mean', 'availability_30':'mean', 'ROI':'mean'}).reset_index()
    top_neighbourhoods = df.groupby('neighbourhood_cleansed')['ROI'].mean().nlargest(10).index
    df = df_grouped[df_grouped['neighbourhood_cleansed'].isin(top_neighbourhoods)].sort_values('Month')

    
    df['neighbourhood_cleansed'] = neighborhood_encoder.inverse_transform(df['neighbourhood_cleansed'])
    df['Month'] = month_encoder.inverse_transform(df['Month'])
    
    # Top 10 Neighbourhood
    fig1 = px.bar(df, 'neighbourhood_cleansed', y = 'ROI', title = 'Top 10 Neighbourhood based on ROI', labels={'x': 'Neighborhood', 'y': 'Monthly Revenue'})
    
    # Monthly Revenue
    fig2 = px.line(df,x='Month', y = 'ROI', title= 'Monthly revenue', color = 'neighbourhood_cleansed')
    
    # Price Range
    fig3 = px.bar(df, x = 'neighbourhood_cleansed', y = 'price', title='Price range', color = 'Month')
    
    # Boking trend
    fig4 = px.bar(x = df['neighbourhood_cleansed'], y = (30 - df['availability_30']), title='Booking trend')
    
    return fig1, fig2, fig3, fig4

def Plotly_graph_for_prediction(df_month, df_noMonth):
    
    neighborhood_encoder = transformations['label_encoders']['neighbourhood_cleansed']
    month_encoder = transformations['label_encoders']['Month']
    
    df_noMonth['recommended_neighborhood'] = neighborhood_encoder.inverse_transform(df_noMonth['recommended_neighborhood'])
    df_month['recommended_neighborhood'] = neighborhood_encoder.inverse_transform(df_month['recommended_neighborhood'])
    df_month['Month'] = month_encoder.inverse_transform(df_month['Month'])
    
    fig1 = px.bar(df_noMonth, 'recommended_neighborhood', y = 'predicted_ROI', title = 'Top 10 Neighbourhood based on ROI', labels={'x': 'Neighborhood', 'y': 'Monthly Revenue'})
    
    fig2 = px.line(df_month ,x='Month', y = 'predicted_ROI', title= 'Monthly revenue', color = 'recommended_neighborhood')
    
    fig3 = px.bar(df_month ,x='recommended_neighborhood', y = 'availability_30', title= 'Booking Trend', color = 'Month')
    
    fig4 = px.line(df_month, x = 'Month', y = 'price', title='Price Range', color='recommended_neighborhood')
    
    return fig1, fig2, fig3, fig4
    
def generate_graphs(df_data, index):
    # Create a copy to avoid modifying the original DataFrame
    df = df_data.copy()

    # Transform encoded columns
    if 'neighbourhood_cleansed' in df.columns:
        df['neighbourhood_cleansed'] = neighborhood_encoder.inverse_transform(df['neighbourhood_cleansed'])

    # Inverse transform numerical columns
    scaler = transformations['scaler']
    df[numerical_cols] = scaler.inverse_transform(df[numerical_cols])

    # KPI 1: Count of unique locations
    total_count = len(df[['latitude', 'longitude']].drop_duplicates())

    # KPI 2: Most common neighborhood
    first_neighborhood = df['neighbourhood_cleansed'].value_counts().idxmax()

    # KPI 3: Total number of reviews
    total_reviews = df['number_of_reviews'].sum()

    # KPI 4: Average minimum and maximum nights
    avg_minimum_nights = df['minimum_nights'].mean()
    avg_maximum_nights = df['maximum_nights'].mean()

    # Bar Chart: Average price by neighborhood
    bar_chart = px.bar(
        df.groupby('neighbourhood_cleansed').agg({'price': 'mean'}).reset_index(),
        x='neighbourhood_cleansed',
        y='price',
        title="Count of Neighborhoods and Average Price",
        labels={'price': 'Average Price'}
    )

    # Line Chart: Average price over months
    line_chart = px.line(
        df.groupby('Month').agg({'price': 'mean'}).reset_index(),
        x='Month',
        y='price',
        title="Average Price Over Months",
        labels={'price': 'Average Price'}
    )

    # Map Visualization: Neighborhood locations
    map_chart = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="neighbourhood_cleansed",
        hover_data=["price", "number_of_reviews"],
        mapbox_style="open-street-map",
        zoom=10,
    )

    # Adjust map layout
    map_chart.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True
    )

    # Bar Chart: Host response and acceptance rates
    host_chart = px.bar(
        df.groupby('neighbourhood_cleansed').agg({
            'host_response_rate': 'mean',
            'host_acceptance_rate': 'mean'
        }).reset_index(),
        x='neighbourhood_cleansed',
        y=['host_response_rate', 'host_acceptance_rate'],
        barmode='group',
        title="Host Response and Acceptance Rates",
        labels={'value': 'Percentage'}
    )

    # Return graphs and KPIs
    return {
        'total_count': total_count,
        'first_neighborhood': first_neighborhood,
        'total_reviews': total_reviews,
        'avg_minimum_nights': avg_minimum_nights,
        'avg_maximum_nights': avg_maximum_nights,
        'bar_chart': bar_chart.to_json(),
        'line_chart': line_chart.to_json(),
        'map_chart': map_chart.to_json(),
        'host_chart': host_chart.to_json()
    }



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/price')
def price():
    return redirect("http://localhost:8501", code=302)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/Dashboard', methods=['GET', 'POST'])
def DashBoard():
    if request.method == 'GET':
        index = "GET"
        # Generate graphs for the full dataset
        graphs = generate_graphs(df_final, index)
        unique_neighborhoods = df_final['neighbourhood_cleansed'].unique()  # Populate dropdown options
        unique_neighborhoods = neighborhood_encoder.inverse_transform(unique_neighborhoods)
        return render_template('dashboard.html', graphs=graphs, unique_neighborhoods=unique_neighborhoods)

    if request.method == 'POST':
        index = "POST"
        filtered_df_post = df_final.copy()

        # Get filter inputs from the form
        neighbourhood = request.form.get('neighbourhood')
        host_id = request.form.get('host_id')

        # Apply filters
        if neighbourhood and neighbourhood != "All":
            encoded_neighbourhood = neighborhood_encoder.transform([neighbourhood])[0]
            filtered_df_post = filtered_df_post[filtered_df_post['neighbourhood_cleansed'] == encoded_neighbourhood]

        if host_id:
            try:
                filtered_df_post = filtered_df_post[filtered_df_post['host_id'] == int(host_id)]
            except ValueError:
                pass  # Handle non-integer host_id gracefully

        # Generate graphs with filtered data
        graphs = generate_graphs(filtered_df_post, index)

        # Populate unique neighborhoods for the dropdown
        unique_neighborhoods = df_final['neighbourhood_cleansed'].unique()
        unique_neighborhoods = neighborhood_encoder.inverse_transform(unique_neighborhoods)
        return render_template('dashboard.html', graphs=graphs, unique_neighborhoods=unique_neighborhoods)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    
    if request.method == 'GET':
        new_df = df_final.copy()
        fig1, fig2, fig3, fig4 = plotly_graph(new_df)
    
        return render_template(
            'recommendation.html',
            fig_json1=fig1.to_json(fig1),
            fig_json2=fig2.to_json(fig2),
            fig_json3=fig3.to_json(fig3),
            fig_json4=fig4.to_json(fig4)
        )
        

    if request.method == 'POST':
        host_id = int(request.form['host_id'])

        # Call your recommendation logic (ensure it returns a DataFrame)
        recommendations = get_recommendation(model, numerical_cols, df_final, host_id, listing_vector)
        Prediction_month, Prediction_NoMonth = prediction_neighbourhood(recommendations, df_final, host_id, model, numerical_cols)
        print(Prediction_month)
        
        scaler = transformations['scaler']
        mean_price = scaler.mean_[5]
        std_price = scaler.scale_[5]
        mean_bookings = scaler.mean_[12]
        std_bookings = scaler.scale_[12]
        mean_ROI = scaler.mean_[-1]
        std_ROI = scaler.scale_[-1]
        
        recommendations.neighbourhood_cleansed = neighborhood_encoder.inverse_transform(recommendations.neighbourhood_cleansed)
        recommendations.mean_ROI = (recommendations.mean_ROI * std_ROI) + mean_ROI
        recommendations.Bookings = 30 - ((recommendations.Bookings * std_bookings) + mean_bookings)
        
        Prediction_month.predicted_ROI = (Prediction_month.predicted_ROI * std_ROI) + mean_ROI
        Prediction_NoMonth.predicted_ROI = (Prediction_NoMonth.predicted_ROI * std_ROI) + mean_ROI
        Prediction_month.availability_30 = 30 -((Prediction_month.availability_30 * std_bookings)+mean_bookings)
        Prediction_month.price = (Prediction_month.price * std_price) + mean_price
        Prediction_NoMonth.availability_30 = 30 -((Prediction_NoMonth.availability_30 * std_bookings)+mean_bookings)
        Prediction_NoMonth.price = (Prediction_NoMonth.price * std_price) + mean_price
        
        
        fig1,fig2,fig3,fig4 = Plotly_graph_for_prediction(Prediction_month,Prediction_NoMonth )

        # Convert recommendations to JSON-serializable forma
        recommendations = recommendations.to_dict(orient='records')
        Prediction_month = Prediction_month.to_dict(orient='records')
        Prediction_NoMonth = Prediction_NoMonth.to_dict(orient='records')

        return render_template(
            'recommendation.html',
            recommendations=recommendations,
            Prediction_month = Prediction_NoMonth,
            fig_json1=fig1.to_json(fig1),
            fig_json2=fig2.to_json(fig2),
            fig_json3=fig3.to_json(fig3),
            fig_json4=fig4.to_json(fig4)
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)
