from __future__ import annotations

import pandas as pd
import logging
import sys
import pendulum
from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator


log = logging.getLogger(__name__)

BUCKET = 'ex_de_data'
LOGICAL_MONTH = "{{ logical_date.strftime('%Y-%m') }}"

# List of files need to be executed
FILES = [
    {"name": "final_dataset", "csv": "final_dataset.csv", "parquet": "final_data.parquet", "bq_table": "data_final"},
    {"name": "stock_data", "csv": "stock_data_YF.csv", "parquet": "stock_data.parquet", "bq_table": "stock_data"},
    {"name": "technical_indicators", "csv": "technical_indicators.csv", "parquet": "technical_indicators.parquet", "bq_table": "technical_indicators"},
    {"name": "ticker_name_list", "csv": "ticker_name_list.csv", "parquet": "ticker_name_list.parquet", "bq_table": "ticker_name_list"},
]

@dag(
    schedule="0 1 1 * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=True,
    max_active_runs=1,
    tags=["example"],
)

def stock_data_pipline():

    # Function to read CSV and save as Parquet
    def save_as_parquet(src_path, parquet_name):
        data = pd.read_csv(src_path)
        data.to_parquet(parquet_name)

    for file in FILES:
        @task(task_id=f"get_{file['name']}")
        def get_data(file=file):
            save_as_parquet(f"/opt/airflow/dags/data/{file['csv']}", file['parquet'])

        upload_file = LocalFilesystemToGCSOperator(
            task_id=f"upload_{file['name']}",
            src=file['parquet'],
            dst=file['parquet'],
            bucket=BUCKET,
        )
    

        load_to_bq = GCSToBigQueryOperator(
            task_id=f"load_{file['name']}_to_bq",
            bucket=BUCKET,
            source_objects=[file['parquet']],
            destination_project_dataset_table=f"sample_warehouse.{file['bq_table']}",
            source_format="PARQUET",
            write_disposition="WRITE_APPEND",
        )
        
    
        # Set dependencies
        get_data_task = get_data()
        get_data_task >> upload_file >> load_to_bq 

    
stock_data_pipline()
