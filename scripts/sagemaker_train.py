"""
sagemaker_train.py — Launch a SageMaker training job

What it does:
  1. Packages src/ code
  2. Uploads to S3
  3. Launches SKLearn estimator on ml.m5.xlarge
  4. SageMaker downloads data from S3
  5. Runs train.py
  6. Uploads model.tar.gz back to S3

Run:
  python scripts/sagemaker_train.py

Cost:
  ml.m5.xlarge = ~$0.23/hr
  Expected training time: 20-30 mins
  Estimated cost: ~$0.10
"""

import os
import sys
import boto3
import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from dotenv import load_dotenv

load_dotenv()

BUCKET        = os.getenv('S3_BUCKET',          'bucke-name')
ROLE_ARN      = os.getenv('SAGEMAKER_ROLE_ARN',  '')
REGION        = os.getenv('AWS_REGION',          'us-east-1')
INSTANCE_TYPE = 'ml.m5.xlarge'
JOB_PREFIX    = 'fraud-detection'


def run_training_job():

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )

    if not ROLE_ARN:
        raise ValueError(
            "SAGEMAKER_ROLE_ARN not set. Add it to .env file."
        )

    # S3 paths
    data_uri   = f's3://{BUCKET}/data'
    output_uri = f's3://{BUCKET}/model'
    code_uri   = f's3://{BUCKET}/code'

    print(f"{'='*55}")
    print(f"  Launching SageMaker Training Job")
    print(f"{'='*55}")
    print(f"  Instance  : {INSTANCE_TYPE}")
    print(f"  Data      : {data_uri}")
    print(f"  Code      : {code_uri}")
    print(f"  Output    : {output_uri}")
    print(f"  Role      : {ROLE_ARN}")
    print(f"{'='*55}\n")

    # Upload latest source code to S3
    print("Uploading source code to S3...")
    s3 = boto3.client('s3', region_name=REGION)
    for fname in ['train.py', 'preprocess.py', 'evaluate.py']:
        local_path = os.path.join('src', fname)
        s3_key     = f'code/{fname}'
        s3.upload_file(local_path, BUCKET, s3_key)
        print(f"  Uploaded {fname} → s3://{BUCKET}/{s3_key}")

    # Create SKLearn estimator
    estimator = XGBoost(
        entry_point       = 'train.py',
        source_dir        = 'src',
        role              = ROLE_ARN,
        instance_count    = 1,
        instance_type     = INSTANCE_TYPE,
        framework_version = '1.7-1',
        py_version        = 'py3',
        output_path       = output_uri,
        base_job_name     = JOB_PREFIX,
        sagemaker_session = session,
        hyperparameters   = {
            'data-dir': '/opt/ml/input/data/train',
        },
        environment = {
            'MODEL_DIR':             '/opt/ml/model',
            'MLFLOW_TRACKING_URI':   os.getenv('MLFLOW_TRACKING_URI', ''),
        },
    )

    print("\nStarting training job...")
    print("(This will take 20-30 mins — you can monitor in AWS Console)")
    print(f"Monitor: https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs\n")

    estimator.fit(
        inputs = {'train': data_uri},
        wait   = True,
        logs   = True,
    )

    print(f"\n{'='*55}")
    print(f"  Training Complete!")
    print(f"{'='*55}")
    print(f"  Model artifact : {estimator.model_data}")
    print(f"\nNext step: python scripts/sagemaker_deploy.py")

    return estimator


if __name__ == '__main__':
    run_training_job()