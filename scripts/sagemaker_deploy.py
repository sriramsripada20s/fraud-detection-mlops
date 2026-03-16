"""
sagemaker_deploy.py — Deploy trained model to SageMaker endpoint

Run:
  python scripts/sagemaker_deploy.py

Delete endpoint when done (saves cost):
  python scripts/sagemaker_deploy.py delete

Cost:
  ml.t2.medium = ~$0.056/hr
  ALWAYS delete when not testing
"""

import os
import sys
import boto3
import sagemaker
from sagemaker.xgboost.model import XGBoostModel
from dotenv import load_dotenv

load_dotenv()

BUCKET        = os.getenv('S3_BUCKET',          'bucket-name')
ROLE_ARN      = os.getenv('SAGEMAKER_ROLE_ARN',  '')
REGION        = os.getenv('AWS_REGION',          'us-east-1')
ENDPOINT_NAME = 'fraud-detection-endpoint'
INSTANCE_TYPE = 'ml.t2.medium'


def deploy_endpoint(model_s3_uri: str = None):

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )

    if not ROLE_ARN:
        raise ValueError("SAGEMAKER_ROLE_ARN not set in .env")

    # Find latest model artifact if not provided
    if model_s3_uri is None:
        s3       = boto3.client('s3', region_name=REGION)
        resp     = s3.list_objects_v2(Bucket=BUCKET, Prefix='model/')
        tarballs = [
            o['Key'] for o in resp.get('Contents', [])
            if o['Key'].endswith('model.tar.gz')
        ]
        if not tarballs:
            raise FileNotFoundError(
                f"No model.tar.gz found in s3://{BUCKET}/model/\n"
                f"Run scripts/sagemaker_train.py first."
            )
        latest_key   = sorted(tarballs)[-1]
        model_s3_uri = f's3://{BUCKET}/{latest_key}'

    print(f"{'='*55}")
    print(f"  Deploying SageMaker Endpoint")
    print(f"{'='*55}")
    print(f"  Model     : {model_s3_uri}")
    print(f"  Endpoint  : {ENDPOINT_NAME}")
    print(f"  Instance  : {INSTANCE_TYPE} (~$0.056/hr)")
    print(f"{'='*55}\n")
    print("WARNING: Remember to delete endpoint when done!")
    print(f"  python scripts/sagemaker_deploy.py delete\n")

    from sagemaker.xgboost.model import XGBoostModel
    model = XGBoostModel(
        model_data        = model_s3_uri,
        role              = ROLE_ARN,
        entry_point       = 'train.py',
        source_dir        = 'src',
        framework_version = '1.7-1',
        py_version        = 'py3',
        sagemaker_session = session,
    )

    predictor = model.deploy(
        initial_instance_count = 1,
        instance_type          = INSTANCE_TYPE,
        endpoint_name          = ENDPOINT_NAME,
    )

    print(f"\n{'='*55}")
    print(f"  Endpoint Live!")
    print(f"{'='*55}")
    print(f"  Endpoint name : {ENDPOINT_NAME}")
    print(f"  Monitor: https://console.aws.amazon.com/sagemaker/home?region={REGION}#/endpoints")
    print(f"\n  DELETE WHEN DONE:")
    print(f"  python scripts/sagemaker_deploy.py delete")

    return predictor


def delete_endpoint():
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )
    try:
        session.delete_endpoint(ENDPOINT_NAME)
        print(f"Deleted endpoint: {ENDPOINT_NAME}")
        print("No more charges for this endpoint.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'delete':
        delete_endpoint()
    else:
        deploy_endpoint()