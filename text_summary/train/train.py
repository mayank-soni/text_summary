from text_summary.train.params import \
    MODEL_CHECKPOINT, METRIC_NAME\
        , BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from text_summary.data.load_data import load_data_from_disk
from text_summary.data.process import preprocess_function

import tensorflow as tf
from datasets import load_metric
from transformers import TFAutoModelForSeq2SeqLM, AdamWeightDecay

# from transformers.keras_callbacks import KerasMetricCallback
# from tensorflow.keras.callbacks import TensorBoard


def train():
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT, from_pt=True)
    metric = load_metric(METRIC_NAME)

    optimizer = AdamWeightDecay(learning_rate=LEARNING_RATE, weight_decay_rate=WEIGHT_DECAY)
    model.compile(optimizer=optimizer)
    train_dataset, validation_dataset, generation_dataset = prepare_data(model)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    try:
        tensorboard_callback = TensorBoard(log_dir="./summarization_model_save/logs")
        model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
        #callbacks=callbacks)
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def prepare_data(model):
    '''
    Loads data from disk, and prepares data for the model.
    Returns
    '''
    data_t, data_v = load_data_from_disk()


    tokenized_data_t = data_t.map(preprocess_function, batched = True)
    tokenized_data_v = data_v.map(preprocess_function, batched = True)

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
