import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pathlib
import pickle
import argparse
import hypertune
import sklearn
from google.cloud import storage
import datetime
def get_args():
    #'Parses args function, include hyperparameters which needs to be tuned.
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',required=True,type=int,help='n_estimators')
    parser.add_argument('--max_depth',required=True,type=int,help='max_depth')
    parser.add_argument('--min_samples_split',required=True,type=int,help='min_samples_split')
    parser.add_argument('--min_samples_leaf',required=True,type=int,help='min_samples_leaf')
    parser.add_argument('--max_features',required=True,type=str,help='max_features')
    args = parser.parse_args()
    return args
def preprocess_data():
    #To read the csv from the cloud storage, and standardize
    data=pd.read_csv("gs://hpo_vertex-ai/EEG_Eye_State_Classification.csv")
    X = data.drop(['eyeDetection'], axis = 1)
    y = data['eyeDetection']
    for column in X.columns:
        X[column] = X[column]  / X[column].abs().max()
    training, testing, training_labels, testing_labels = train_test_split(X, y, test_size = .25, random_state = 42)
    return training, testing, training_labels, testing_labels
def main():
    args = get_args()
    training, testing, training_labels, testing_labels = preprocess_data()
    model_classifier = RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth,min_samples_split=args.min_samples_split,min_samples_leaf=args.min_samples_leaf,max_features=args.max_features)
    model_classifier.fit(training,training_labels)
    y_pred = model_classifier.predict(testing)
    acc = accuracy_score(testing_labels, y_pred)
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=acc)
    artifact_filename = "model.pkl"
    pickle.dump(model_classifier, open(artifact_filename, "wb"))
    BUCKET = 'hpo_vertex-ai'
    gcs = storage.Client(project="Vertex-ai")
    buck=gcs.bucket(BUCKET)
    blob = buck.blob(artifact_filename)
    blob.upload_from_filename(artifact_filename)

if __name__ == "__main__":
    main()
