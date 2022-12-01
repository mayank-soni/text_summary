import os

#model, metrics, data
MODEL_CHECKPOINT = 'sshleifer/distilbart-cnn-12-6'
DATASET_NAME = 'xsum'
METRIC_NAME = 'rouge'
DATA_PATH = os.path.join(os.getcwd(), 'raw_data')

#data
ARTICLE_COLUMN = 'article'
SUMMARY_COLUMN = 'highlights'

#train
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 1
