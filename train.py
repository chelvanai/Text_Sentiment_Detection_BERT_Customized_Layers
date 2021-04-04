import math
import random
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
from model import SentimentModel
import bert

df = pd.read_csv('train_story_data.csv')
TAG_RE = re.compile(r'<[^>]+>')
text = []


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


sentences = list(df['sentence'])
for sen in sentences:
    text.append(preprocess_text(sen))


def extraction(emotion):
    emotion = emotion[0]
    return emotion


df['Emotion'] = df.loc[:, 'ann1_emotion'].str.split(':')
df['Emotion'] = df['Emotion'].apply(extraction)
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su-', 'Su')
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su+', 'Su')
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('D', 'A')

# Encoding sentiments
label_encoder = preprocessing.LabelEncoder()
df['Emotion'] = label_encoder.fit_transform(df['Emotion'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
le_mapping = {'Emotion': le_name_mapping}
print(df['Emotion'].values)
print(le_mapping)

y = df['Emotion'].values

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


def tokenize_text(text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


tokenized_text = [tokenize_text(txt) for txt in text]
text_with_len = [[text, y[i], len(text)]
                 for i, text in enumerate(tokenized_text)]
random.shuffle(text_with_len)
text_with_len.sort(key=lambda x: x[2])
sorted_sentence_emotions = [(review_lab[0], review_lab[1]) for review_lab in text_with_len][20:]

processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_sentence_emotions, output_types=(tf.int32, tf.int32))
BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,), ()))

print(next(iter(processed_dataset)))

TOTAL_BATCHES = math.ceil(len(sorted_sentence_emotions) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 6

DROPOUT_RATE = 0.2

NB_EPOCHS = 10

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

model.fit(train_data, epochs=NB_EPOCHS)

# -------------------------------------------------------------------------------
# Evaluation process

df = pd.read_csv('test_story_data.csv')
df['Emotion'] = df.loc[:, 'ann1_emotion'].str.split(':')
df['Emotion'] = df['Emotion'].apply(extraction)
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su-', 'Su')
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su+', 'Su')
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('D', 'A')

# Encoding sentiments
label_encoder = preprocessing.LabelEncoder()
df['Emotion'] = label_encoder.fit_transform(df['Emotion'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
le_mapping = {'Emotion': le_name_mapping}
print(le_mapping)

y = df['Emotion'].values        # Test sentiment values

text = []                       # Test sentence values
sentences = list(df['sentence'])
for sen in sentences:
    text.append(preprocess_text(sen))

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


def tokenize_text(text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


tokenized_text = [tokenize_text(txt) for txt in text]
text_with_len = [[text, y[i], len(text)]
                 for i, text in enumerate(tokenized_text)]
random.shuffle(text_with_len)
text_with_len.sort(key=lambda x: x[2])
sorted_sentence_emotions = [(review_lab[0], review_lab[1]) for review_lab in text_with_len][20:]

processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_sentence_emotions, output_types=(tf.int32, tf.int32))
BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,), ()))

TOTAL_BATCHES = math.ceil(len(sorted_sentence_emotions) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

res = model.evaluate(train_data)
print(res)
