import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlowModel

region = 'us-east-1'
sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

role = 'arn:aws:iam::726510512265:role/sage_deploy' 

model_path = 's3://cuna-bebe-326820/my_model/posture_model_5.tar.gz'

tensorflow_model = TensorFlowModel(
    model_data=model_path,
    role=role,
    framework_version='2.11', 
    sagemaker_session=sagemaker.Session()
)


predictor = tensorflow_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium', 
    endpoint_name='posture-v5'
)
