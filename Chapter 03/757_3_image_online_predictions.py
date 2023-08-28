#Install google-cloud-aiplatform
import base64

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =r"E:\service_account_json\vertex-ai-gcp-0a71505b3eae.json"
project_ID= "vertex-ai-gcp" #project_ID can be obtained from GCP dashboard (refer figure 2.35)
model_ID= "4810468924636266496" #Model_ID refer figure 2.29
filename= r"E:\Publication\CHAPTERS\CHAPTER 3\DATA\Batch Pred\FB.jpg"  #source of image to be predicted
location= "us-central1"#Location refer to figure 2.25
api_endpoint= "us-central1-aiplatform.googleapis.com"

def predict_image_classification(
    project_ID= project_ID,
    model_ID= model_ID,
    filename= filename,
    location= location,
    api_endpoint= api_endpoint):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()
    content_encoded = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(content=content_encoded).to_value()
    instances = [instance]
    params= predict.params.ImageClassificationPredictionParams(confidence_threshold=0.5, max_predictions=1).to_value()
    endpoint = client.endpoint_path(project=project_ID, location=location, endpoint=model_ID)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=params)
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

#Calling the function

predict_image_classification(project_ID,model_ID,filename,location,api_endpoint)
