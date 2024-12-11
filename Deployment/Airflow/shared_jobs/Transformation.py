from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, sum as _sum, when


def transform_data(file_path, spark):
    df = spark.read.csv(
                file_path,
                header=True,
                inferSchema=True,
                quote='"',
                escape='"'
            ).repartition(10)
    print('Handling symbols and converting price to float...')
    df = df.withColumn('price', F.regexp_replace('price', '\\$', ''))
    df = df.withColumn('price', F.regexp_replace('price', ',', ''))
    df = df.withColumn('price', F.col('price').cast('double'))
    print('Done!!')


    # Removing text from numberical columns
    columns = ['bathrooms_text']
    for column in columns:
            # Use regexp_extract to extract the numerical part from the string
        df = df.withColumn(column, F.regexp_extract(F.col(column), r'(\d+(\.\d+)?)', 0).cast('double'))
    print('Text from numbers are removed.')


    # Remove the percentage sign and converting to float
    columns = ['host_response_rate', 'host_acceptance_rate']
    for column in columns:
        df = df.withColumn(column, F.regexp_replace(F.col(column), '%', '').cast('double'))
    print('Removing the percentage sign and converted to float')


    print(f'Triming whitespace')
    for column in df.columns:
        df = df.withColumn(column, F.trim(F.col(column)))
    print('Done.')


    # Converting the numerical columns to numerical values
    columns_to_convert = [
        'id',
        'scrape_id',
        'host_id',
        'latitude',
        'longitude',
        'accommodates',
        'bedrooms',
        'beds',
        'minimum_nights',
        'maximum_nights',
        'minimum_minimum_nights',
        'maximum_minimum_nights',
        'minimum_maximum_nights',
        'maximum_maximum_nights',
        'minimum_nights_avg_ntm',
        'maximum_nights_avg_ntm',
        'availability_30',
        'availability_60',
        'availability_90',
        'availability_365',
        'number_of_reviews',
        'number_of_reviews_ltm',
        'number_of_reviews_l30d',
        'calculated_host_listings_count',
        'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms',
        'reviews_per_month'
    ]
    for column in columns_to_convert:
        df = df.withColumn(column, F.col(column).cast('float'))
        
        
    # Filling null values
    average_price_df = df.groupBy('neighbourhood_cleansed', 'property_type', 'accommodates') \
                        .agg(F.avg('price').alias('average_price'))

    df = df.join(average_price_df, 
                on=['neighbourhood_cleansed', 'property_type', 'accommodates'], 
                how='left')

    df = df.withColumn('price', F.coalesce(F.col('price'), F.col('average_price')))

    df = df.drop('average_price')
    df = df.dropna(subset=['price'])
    print('Filled nulled value with average price.')


    # Changing the empty string to None (null value)
    for column in df.columns:
        df = df.withColumn(column, F.when(F.col(column) == '', None).otherwise(F.col(column)))
        

    # Filling beds and bedrooms with average of accommodates and room_type
    average_values = df.groupBy('room_type', 'accommodates') \
        .agg(
            F.avg('beds').alias('avg_beds'),
            F.avg('bedrooms').alias('avg_bedrooms')
        )

    df = df.join(average_values, on=['room_type', 'accommodates'], how='left')


    df = df.withColumn('beds', F.coalesce(F.col('beds'), F.col('avg_beds'))) \
        .withColumn('bedrooms', F.coalesce(F.col('bedrooms'), F.col('avg_bedrooms')))

    df = df.drop('avg_beds', 'avg_bedrooms')
    print('Beds and bedrooms null values are handle.')


    # response_columns = ['host_response_time', 'host_response_rate', 'host_acceptance_rate'] filling response column
    response_columns = ['host_response_time', 'host_response_rate', 'host_acceptance_rate']

    average_response_values = df.agg(
        *[F.avg(column).alias(f'avg_{column}') for column in response_columns]
    )

    # Collecting the average values into a dictionary
    avg_values = average_response_values.first().asDict()

    # Filling missing values in the response columns with the calculated averages
    for column in response_columns:
        df = df.withColumn(column, F.coalesce(F.col(column), F.lit(avg_values[f'avg_{column}'])))
    print('Response column are handle.')

    # Droping unwanted columns
    unwanted_column = []
    df = df.drop('description','host_location', 'host_about', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'host_neighbourhood')
    df = df.fillna({'license': '0'})
    df = df.fillna({'host_response_time':5})


    print('Filling the neighborhood_overview column with the overview from other listings that share the same neighbourhood_cleansed')
    agg_df = df.groupBy("neighbourhood_cleansed") \
                .agg(F.first("neighborhood_overview").alias("overview"))  # Get the first non-null overview for each neighborhood

    # Join the aggregated DataFrame back to the original DataFrame
    df = df.join(agg_df, on="neighbourhood_cleansed", how="left")

    # Filling the null values in neighborhood_overview with the overview from the aggregated DataFrame
    df = df.withColumn(
        "neighborhood_overview",
        F.coalesce(df["neighborhood_overview"], df["overview"])
    )
    df = df.drop("overview")
    print('Neighbourhood column are handled.')


    print('filling availability and retings')
    df = df.withColumn(
        "has_availability",
        F.when(
            (F.col("availability_30") == 0) & 
            (F.col("availability_60") == 0) & 
            (F.col("availability_90") == 0) & 
            (F.col("availability_365") == 0), "f"
        ).otherwise("t")
    )
    df = df.fillna({'host_is_superhost':'f'})
    df = df.fillna({'neighborhood_overview': 'UNKNOWN'})
    df = df.drop('first_review','last_review')

    df = df.fillna({
        "review_scores_rating": 0.0,
        "review_scores_accuracy": 0.0,
        "review_scores_cleanliness": 0.0,
        "review_scores_checkin": 0.0,
        "review_scores_communication": 0.0,
        "review_scores_location": 0.0,
        "review_scores_value": 0.0,
        "reviews_per_month": 0.0
    })
    print('Success.')



    # Droping columns which have more than 15% null values and than droping null
    print('>15% null values handling')
    total_rows = df.count()
        
    null_counts = df.select([
            _sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in df.columns
        ])
        
    null_percentages = {
            col_name: (null_counts.collect()[0][col_name] / total_rows)
            for col_name in df.columns
        }

    columns_to_keep = [
            col_name for col_name, null_percentage in null_percentages.items()
            if null_percentage <= 0.15
        ]
        
    df_cleaned = df.select(columns_to_keep)
        
    df_cleaned = df_cleaned.dropna()
    print("Handled...")

    df_cleaned.coalesce(1).write.csv('/opt/Data/cleanedData.csv', header=True)


if __name__=='__main__':
    
    spark = SparkSession.builder.appName("Tranformation_Cleaning").getOrCreate()

    file_path = '/opt/Data/listings.csv'
    
    transform_data(file_path, spark)

