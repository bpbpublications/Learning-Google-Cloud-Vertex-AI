#Install google-cloud-aiplatform
import os
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
#Working code for hyperparameters

def online_custom_models(project,endpoint_id,instance_dict,location,api_endpoint):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    print(endpoint)
    response = client.predict(endpoint=endpoint, instances=instances)
    predictions = response.predictions
    print(predictions)


#For GCP authentication.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =r"E:\service_account_json\vertex-ai-gcp-1-a3973ca6a1fe.json"

project_ID="vertex-ai-gcp-1"
endpoint_ID="5530565477945835520"
location = "us-central1"
api_endpoint="us-central1-aiplatform.googleapis.com"
#Calling the function

inputs=[4287.69,3997.44,4260,4121.03,4333.33,4616.41,4088.72,4638.46,4212.31,4226.67,4167.69,4274.36,4597.95,4350.77]
online_custom_models(project_ID,endpoint_ID,inputs,location,api_endpoint)
