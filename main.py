from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from datetime import date
# from typing import Union
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# http://127.0.0.1:8000/predict?category=sports&from=2022-11-01&language=en
@app.get("/predict")
def predict(date_from: str,             # 2022-11-01 (YYYY-MM-DD)
            keywords: str,              # e.g world cup / bitcoin
            category: str = 'sports'    # sports

):
    url = "https://newsapi.org/v2/top-headlines"
    params = {"apiKey": os.environ['newsapi_API_KEY'], 'category':f'{category}', 'language':'en', 'q':f'{keywords}', 'from':f'{date_from}'}
    response = requests.get(url, params=params)
    return response.json()
