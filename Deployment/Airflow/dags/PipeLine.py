from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from Extraction import download_last_12_months
from transformation import transform_data
from merging import merge_csv_files
from event_datacollector import Event_data
from datetime import datetime
import os
import requests
from processing import Processing_for_model
from Transformation_XGBoost import main_transformation
from random import randint


def machine_leaning(model_name):
    rmse = randint(5, 15)
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
    'DataPipeline',
    start_date=datetime(year=2024, month=11, day=25),
    schedule_interval='@monthly',
    description='ETL Pipeline',
    tags=['ETL','Python'],
    catchup=False,
) as dag:

    # Task 1: Perform Addition
    Download_Monthly_Data = PythonOperator(
        task_id='Download_Monthly_Data',
        python_callable=download_last_12_months,
    )
    
    Event_data_Collector = PythonOperator(
        task_id='Event_Data_Collector',
        python_callable=Event_data,
        op_kwargs={
            'filename': '/opt/Data'
        },
    )


    Merge_task = PythonOperator(
        task_id='merge_csv_files_task',
        python_callable=merge_csv_files,
        op_kwargs={
            'base_folder': '/opt/Data',
            'output_path': '/opt/Data/merged_output.csv',
        },
    )
    
    Transformation = PythonOperator(
        task_id='Data_Cleaning',
        python_callable=transform_data,
        op_kwargs={
            'file_path': '/opt/Data/merged_output.csv',
            'output_path': '/opt/Data/CleanData.csv',
        },
    )
    
    Processing = PythonOperator(
        task_id='Transformation_Feature_ENgineering',
        python_callable=Processing_for_model,
        op_kwargs={
            'inputfilepath': '/opt/Data/CleanData.csv',
            'outputfilepath': '/opt/Data',
        },
    )
    
    Preprocessing_for_predictive_model = PythonOperator(
        task_id='Feature_Engineering_Predictive_model',
        python_callable=main_transformation,
        op_kwargs={
            'inputfilepath': '/opt/Data/merged_output.csv',
            'outputfilepath': '/opt/Data',
        },
    )
    
    Recommendation_Engine = PythonOperator(
        task_id='Recommendation_Model',
        python_callable=machine_leaning,
        op_kwargs={'model_name': 'LSTM'})
    
    
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
    [Download_Monthly_Data, Event_data_Collector] >> Merge_task >> Transformation >> [Processing, Preprocessing_for_predictive_model]
    
    Preprocessing_for_predictive_model >> [XGBOOST, LSTM, CNN] >> Best_Model
    
    Processing >> Recommendation_Engine
    
    Transformation >> [Recommendation_Engine, Best_Model]
    
    

    