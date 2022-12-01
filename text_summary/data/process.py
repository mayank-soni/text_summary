from text_summary.train.params import \
    MODEL_CHECKPOINT, ARTICLE_COLUMN, SUMMARY_COLUMN

from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq


def preprocess_function(data):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    inputs = [doc for doc in data[ARTICLE_COLUMN]]
    tokenized_data = tokenizer(text=inputs,
                               truncation=True,
                               text_target=data[SUMMARY_COLUMN])
    return tokenized_data

def collate_date(data, model)
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           model=model,
                                           return_tensors="tf")
    generation_data_collator = DataCollatorForSeq2Seq(tokenizer,
                                                      model=model,
                                                      return_tensors="tf",
                                                      pad_to_multiple_of=128)
