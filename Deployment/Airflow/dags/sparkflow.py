from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from Extraction import download_last_12_months
from datetime import datetime
import os
import random
import requests



def machine_leaning(model_name):
    rmse = random.randint(5, 15)
    print(f"Model: {model_name}, RMSE: {rmse}")
    return {"model": model_name, "rmse": rmse}

def choose_best_model(**kwargs):
    ti = kwargs['ti']
    models = ['XGBOOST', 'LSTM', 'CNN']
    model_rmses = [ti.xcom_pull(task_ids=f"{model}_Model") for model in models]
    
    best_model = min(model_rmses, key=lambda x: x['rmse'])
    print(f"Best Model: {best_model['model']} with RMSE: {best_model['rmse']}")
    return best_model


with DAG(
    'Spark_pipeline',
    start_date=datetime(year=2024, month=11, day=25),
    schedule_interval='@monthly',
    description='ETL Pipeline',
    tags=['ETL','Spark', 'data-ingestion'],
    catchup=False,
) as dag:

    # Task 1: Perform Addition
    Download_Monthly_Data = PythonOperator(
        task_id='Download_Monthly_Data',
        python_callable=download_last_12_months,
    )


    Merging = SparkSubmitOperator(
        task_id="Merging_Files",
        application="/opt/shared_jobs/pyspark_job.py",
        conn_id="spark_default", 
        verbose=True,
        # application_args=[
        #     "--base_folder", "/opt/Data",  # Passing the base folder dynamically if needed
        #     "--output_path", "/opt/Data/merged.csv"],
    )
    
    run_spark_job = BashOperator(
        task_id='run_spark_job',
        bash_command="""
        /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        --executor-memory 4G \
        --driver-memory 4G \
        /opt/shared_jobs/pyspark_job.py
        """
        )
    
    Transformation = SparkSubmitOperator(
        task_id="Transformation",
        application="/opt/shared_jobs/Transformation.py",
        conn_id="spark_default",  
        verbose=True, 
        conf={
        "spark.executor.memory": "4g",
        "spark.driver.memory": "4g",
        "spark.executor.cores": "4",
        "spark.sql.shuffle.partitions": "10",
    }
        )
    
    Preprocessing_for_predictive_model = PythonOperator(
        task_id='Feature_Engineering',
        python_callable=machine_leaning,
        op_kwargs={
            'inputfilepath': '/opt/Data/output/CleanData.csv',
            'outputfilepath': '/opt/Data/output',
        },
    )
    
    Recommendation_Engine = PythonOperator(
        task_id='Recommendation_System',
        python_callable=machine_leaning,
    )
    
    XGBOOST = PythonOperator(
            task_id = 'XGBOOST_Model',
                             python_callable=machine_leaning,
                             op_kwargs={'model_name': 'XGBOOST'}
                             )
    
    LSTM = PythonOperator(task_id = 'LSTM_Model',
                             python_callable=machine_leaning,
                             op_kwargs={'model_name': 'LSTM'})
    
    CNN = PythonOperator(task_id = 'CNN_Model',
                             python_callable=machine_leaning,
                             op_kwargs={'model_name': 'CNN'},)
    
    Best_Model = PythonOperator(
        task_id = 'Choose_Best_Model',
        python_callable=choose_best_model,
    )

    # Define Task Dependencies
    Download_Monthly_Data >> Merging >> Transformation >> [Preprocessing_for_predictive_model , Recommendation_Engine]
    
    Preprocessing_for_predictive_model >> [XGBOOST, LSTM, CNN] >> Best_Model
    

    