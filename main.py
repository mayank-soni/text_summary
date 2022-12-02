from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from transformers import AutoTokenizer, pipeline, TFAutoModelForSeq2SeqLM
# from tensorflow.python.keras.saving import hdf5_format

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {'Hello': 'World'}

#http://127.0.0.1:8000/predict?category=sports&language=en&q=worldcup
@app.get('/predict')
def predict(keywords: str,              #e.g. world cup
            category: str = 'sports'
):
    url = 'https://newsdata.io/api/1/news'
    params = {"apikey": os.environ['newsData_API_key'], "category":f"{category}", "language":"en", "q":f"{keywords}"}
    response = requests.get(url, params=params)
    #return response.json()
    prediction = response.json()
    # return prediction
    articles = []
    for item in prediction['results'][0:1]:
        if item['content'] != None :
            articles.append(item['content'])

    summaries = []
    for article in articles:
        summaries.append(summarization(article))
    return summaries
    # final = {articles[i]: summaries[i] for i in range(len(articles))}

    # return final

# load model
new_model = TFAutoModelForSeq2SeqLM.from_pretrained('trained_model')
new_model.summary()

# load summary of sports articles
def summarization(item):
    # return 'hello'
    model_checkpoint = 'sshleifer/distilbart-cnn-12-6'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized = tokenizer(item, return_tensors='np')
    return tokenized
    # out = new_model.generate(**tokenized, max_length=128)

    # with tokenizer.as_target_tokenizer():
    #     return(tokenizer.decode(out[0]))
