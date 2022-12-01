from text_summary.train.params import \
    MODEL_CHECKPOINT, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from text_summary.data.load_data import load_data_from_disk
from text_summary.data.process import preprocess_function
from text_summary.train.metrics import metric_fn

# import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AdamWeightDecay,\
    DataCollatorForSeq2Seq, AutoTokenizer

from transformers.keras_callbacks import KerasMetricCallback
from tensorflow.keras.callbacks import TensorBoard


def train():
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT, from_pt=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    optimizer = AdamWeightDecay(learning_rate=LEARNING_RATE, weight_decay_rate=WEIGHT_DECAY)
    model.compile(optimizer=optimizer)
    train_dataset, validation_dataset, generation_dataset = prepare_data(model, tokenizer)
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    # try:
    #     tensorboard_callback = TensorBoard(log_dir="./summarization_model_save/logs")
    #     model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
    #     #callbacks=callbacks)
    # # Currently, memory growth needs to be the same across GPUs
    #     for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    #     logical_gpus = tf.config.list_logical_devices('GPU')
    #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    # except RuntimeError as e:
    #     # Memory growth must be set before GPUs have been initialized
    #     print(e)
    tensorboard_callback = TensorBoard(log_dir="./summarization_model_save/logs")

    metric_callback = KerasMetricCallback(
        metric_fn, eval_dataset=generation_dataset, predict_with_generate=True, use_xla_generation=True
    )

    callbacks = [metric_callback, tensorboard_callback] #, push_to_hub_callback

    model.fit(
        train_dataset, validation_data=validation_dataset, epochs=1, callbacks=callbacks)

    return model


def prepare_data(model, tokenizer):
    '''
    Loads data from disk, and prepares data for the model.
    Returns
    '''
    data_t, data_v = load_data_from_disk()

    tokenized_data_t = data_t.map(preprocess_function, batched = True)
    tokenized_data_v = data_v.map(preprocess_function, batched = True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
    generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

    train_dataset = model.prepare_tf_dataset(
        tokenized_data_t,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    validation_dataset = model.prepare_tf_dataset(
        tokenized_data_v,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    generation_dataset = model.prepare_tf_dataset(
        tokenized_data_v,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=generation_data_collator
    )
    return train_dataset, validation_dataset, generation_dataset


def predict():
    _, _, predict_dataset = prepare_data(model)
    out = model.generate(**tokenized, max_length=128)


if __name__ == '__main__':
    model = train()
