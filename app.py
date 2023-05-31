
import joblib
import json
import pandas as pd
import numpy
#import pytest
from fastapi import FastAPI
from mangum import Mangum


#Run pytest Unit tests here
#pytest


app_logistic = FastAPI()

@app_logistic.get("/")  # The INDEX (main) page of the EndPoint will return the JSON file below (no INPUT needed)
def index():
    return {"message": "Hello, new user"}


@app_logistic.post("/predict")
# The function below tells the Endpoint how to calculate predictions from the JSON input received (here a tweet features in format for LSTM)
def process_post_request(data_as_json):

    # Load existing model to do predict with it
    model_logistic = joblib.load("model_logistic_shorty.pkl")

    # Make the received input JSON into a dictionary
    data = json.loads(data_as_json)

    # Make the dictionary into a dataframe
    df = pd.DataFrame.from_dict(data)

    # Model's probability prediction
    sentiment = model_logistic.predict(df)

    return {  # "message": "Code 200 - POST request processed successfully",
        "sentiment": sentiment.item()}


handler = Mangum(app_logistic)