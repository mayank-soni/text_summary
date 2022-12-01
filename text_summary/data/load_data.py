from text_summary.train.params import \
    DATA_PATH

import os

from datasets import load_from_disk

def load_data_from_disk():
    train_data_path = os.path.join(DATA_PATH, 'train_data')
    validation_data_path = os.path.join(DATA_PATH, 'validation_data')

    raw_datasets_t = load_from_disk(train_data_path)
    raw_datasets_v = load_from_disk(validation_data_path)
    return raw_datasets_t, raw_datasets_v
