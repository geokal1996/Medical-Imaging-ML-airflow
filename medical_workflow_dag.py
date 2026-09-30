from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys


# Directory containing the pipeline scripts.
# It can be overridden using the MEDICAL_PIPELINE_DIR environment variable.
PIPELINE_DIR = os.environ.get(
    "MEDICAL_PIPELINE_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PIPELINE_DIR)


from data_ingestion import ingest_dicom_files
from data_preprocessing import preprocess_dicom
import model_training


default_args = {
    'owner': 'giorgos',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 27),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


dag = DAG(
    'medical_workflow_pipeline',
    default_args=default_args,
    description='Automated ML pipeline for medical imaging',
    schedule=timedelta(days=1),
    catchup=False,
)


def run_dicom_processing():
    ingest_dicom_files(
        input_dir=os.path.join(PIPELINE_DIR, "data", "raw_dicom"),
        output_dir=os.path.join(PIPELINE_DIR, "data", "anonymized_dicom"),
    )


def run_preprocessing():
    preprocess_dicom(
        input_dir=os.path.join(PIPELINE_DIR, "data", "anonymized_dicom"),
        output_dir=os.path.join(PIPELINE_DIR, "data", "processed_dicom"),
    )


def run_training():
    # Run model training from the pipeline working directory.
    os.chdir(PIPELINE_DIR)
    model_training.main()


dicom_task = PythonOperator(
    task_id='process_dicom',
    python_callable=run_dicom_processing,
    dag=dag,
)


preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=run_preprocessing,
    dag=dag,
)


train_task = PythonOperator(
    task_id='train_model',
    python_callable=run_training,
    dag=dag,
)


dicom_task >> preprocess_task >> train_task