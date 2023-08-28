import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =r"" # Full path of the service account key (Json file)
project_ID= "" #Project_ID of the current project
endpoint_id= "" #End point of the model
content_path= "" #Full path of the content file
location= "us-central1"
api_endpoint= "us-central1-aiplatform.googleapis.com"

def predict_text_sentiment_analysis_sample(
    project_ID: str,
    endpoint_id: str,
    content_path: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com"):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    for path in os.listdir(content_path):
        with open(os.path.join(content_path, path)) as f:
            content1=f.readlines()
        instance = predict.instance.TextSentimentPredictionInstance(content=str(content1[0]),).to_value()
        instances = [instance]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        endpoint = client.endpoint_path(project=project_ID, location=location, endpoint=endpoint_id)
        response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
        predictions = response.predictions
        for prediction in predictions:
            print(" prediction:", dict(prediction))
#Calling the function to predict the sentiment
predict_text_sentiment_analysis_sample(project_ID,endpoint_id,content_path)
