from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

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

# load model 1 (base model)
model_checkpoint = 'sshleifer/distilbart-cnn-12-6'
new_model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, from_pt = 'True')
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# load summary of model 1 (base model)
def summarization(tokenized):
    out = new_model.generate(**tokenized, max_length=150)
    with tokenizer.as_target_tokenizer():
        return(tokenizer.decode(out[0]))

# load model 2 (trained model)
new_model2 = TFAutoModelForSeq2SeqLM.from_pretrained('trained_model')

# load summary of model 2 (trained model)
def summarization2(tokenized):
#     tokenizer2 = AutoTokenizer.from_pretrained('trained_model')
    # tokenized2 = tokenizer(item, return_tensors='np')
    out = new_model2.generate(**tokenized, max_length=150)
    with tokenizer.as_target_tokenizer():
        return(tokenizer.decode(out[0]))

#http://127.0.0.1:8000/predict?category=sports&language=en&q=worldcup
@app.get('/predict')
def predict(keywords: str):              #e.g. world cup

    url = 'https://newsdata.io/api/1/news'
    params = {"apikey": os.environ['newsData_API_key'], "category":"sports", "language":"en", "q":f"{keywords}"}
    response = requests.get(url, params=params)
    #return response.json()
    prediction = response.json()

    # return prediction
    articles = []
    for item in prediction['results'][0:1]:
        if item['content'] != None :
            articles.append(item['content'])

    summaries = []
    summaries2 = []
    for article in articles:
        tokenized = tokenizer(article, return_tensors='np')
        summaries.append(summarization(tokenized))
        summaries2.append(summarization2(tokenized))

    # return {'summary': summaries[0]}
    final = {'article': articles[0],
             'summary': summaries[0].lstrip('</s><s> ').rstrip('<s> ').rstrip('<pad>').rstrip('</s'),
             'summary2': summaries2[0].replace('\n',' ').strip('</s>').rstrip('<s> ').rstrip('<pad>').rstrip('</s')}

    return final
