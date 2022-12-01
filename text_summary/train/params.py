import os

MODEL_CHECKPOINT = 'sshleifer/distilbart-cnn-12-6'
DATASET_NAME = 'xsum'
METRIC_NAME = 'rouge'
DATA_PATH = os.path.join(os.getcwd(), 'raw_data')

ARTICLE_COLUMN = 'article'
SUMMARY_COLUMN = 'highlights'
