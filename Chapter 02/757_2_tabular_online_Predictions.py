from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os

#Install google-cloud-aiplatform
def online_pred_tabular(project,model_id,instance_dict,location,api_endpoint):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(project=project, location=location, endpoint=model_id)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    predictions = response.predictions
    for prediction in predictions:print(" prediction:", dict(prediction))


#For GCP authentication. Provide full path of the json file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =""
project_id="vertex-ai-gcp" #project_ID can be obtained from GCP dashboard (refer figure 2.35)
model_id="2047967950081622016" #Model_ID refer figure 2.29
location = "us-central1" #Location refer to figure 2.25
api_endpoint="us-central1-aiplatform.googleapis.com"
#Inputs in Json format
inputs = {"BMI":"16.6","Smoking":"Yes","AlcoholDrinking":"No","Stroke":"No","PhysicalHealth":"3","MentalHealth":"30","DiffWalking":"No","Sex":"Female","AgeCategory":"55-59","Race":"White",Diabetic":"Yes","PhysicalActivity":"Yes","GenHealth":"Very good","SleepTime":"5","Asthma":"Yes","KidneyDisease":"No","SkinCancer":"Yes"}

#Calling the function
online_pred_tabular(project_id, model_id,inputs,location,api_endpoint)


