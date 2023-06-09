
#import os
import json
import pandas as pd
import sklearn
import joblib
#import numpy
import fastapi
from fastapi import FastAPI
import mangum
from mangum import Mangum

app_logistic = FastAPI()

@app_logistic.get("/")  # The INDEX (main) page of the EndPoint will return the JSON file below (no INPUT needed)
def index():
    return {"message": "Hello to you, new user"}


@app_logistic.post("/predict")
# The function below tells the Endpoint how to calculate predictions from the JSON input received (here a tweet features in format for LSTM)
def process_post_request(data_as_json):

    # Load existing model to do predict with it
    #model_path = os.getenv("MODEL_PATH")
    #model_logistic = joblib.load(model_path)

    model_logistic = joblib.load("model_logistic_shorty_word2vec.pkl")
    #print(joblib.__version__)

    # Make the received input JSON into a dictionary
    data = json.loads(data_as_json)

    # Make the dictionary into a dataframe
    df = pd.DataFrame.from_dict(data)

    # Model's probability prediction
    sentiment = model_logistic.predict(df)

    return { "Status": "200 - POST request SUCCESS",
             "Prediction": "0= Negative and 1= Positive",
             "Tweet_sentiment": sentiment.item()
            }


handler = Mangum(app_logistic)