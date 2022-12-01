from text_summary.train.params import \
    MODEL_CHECKPOINT, DATASET_NAME, METRIC_NAME
from text_summary.data.load_data import load_data_from_disk
from text_summary.data.process import preprocess_function

from transformers import AutoTokenizer
from datasets import load_metric

data_t, data_v = load_data_from_disk()
metric = load_metric(METRIC_NAME)

tokenized_data_t = data_t.map(preprocess_function, batched = True)
tokenized_data_v = data_v.map(preprocess_function, batched = True)
