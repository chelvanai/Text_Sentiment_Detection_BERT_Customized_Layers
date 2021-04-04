import math
import random
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from model import SentimentModel
import bert

# Tokenize
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 6

DROPOUT_RATE = 0.2

NB_EPOCHS = 8

model = SentimentModel(vocabulary_size=VOCAB_LENGTH,
                       embedding_dimensions=EMB_DIM,
                       cnn_filters=CNN_FILTERS,
                       dnn_units=DNN_UNITS,
                       model_output_classes=OUTPUT_CLASSES,
                       dropout_rate=DROPOUT_RATE)

if OUTPUT_CLASSES == 2:
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
else:
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["sparse_categorical_accuracy"])

latest = tf.train.latest_checkpoint('./new_weights')
model.load_weights(latest)

# model.load_weights('./weights/base_model_weights')
model.build(None, None)


# print(model.summary())


def encode_sentence(sent):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))


def get_prediction(sentence):
    tokens = encode_sentence(sentence)
    inputs = tf.expand_dims(tokens, 0)

    output = model(inputs, training=False)

    print(output)
    print([np.argmax(output)])


print(model.summary())
get_prediction(
    "We will not open the door,\" cried they, \"you are not our mother.")
