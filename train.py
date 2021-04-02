import math
import random
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
from tensorflow.keras import layers
import bert

df = pd.read_csv('train_story_data.csv')

TAG_RE = re.compile(r'<[^>]+>')

# pre process sentences from dataset and make as list
text = []


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


sentences = list(df['sentence'])
for sen in sentences:
    text.append(preprocess_text(sen))


# pre process annotation of emotion
def extraction(emotion):
    emotion = emotion[0]
    return emotion


df['Emotion'] = df.loc[:, 'ann1_emotion'].str.split(':')
df['Emotion'] = df['Emotion'].apply(extraction)

df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su-', 'Su')
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('Su+', 'Su')
df.loc[:, 'Emotion'] = df.loc[:, 'Emotion'].replace('D', 'A')

print(df.head())

label_encoder = preprocessing.LabelEncoder()

df['Emotion'] = label_encoder.fit_transform(df['Emotion'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
le_mapping = {'Emotion': le_name_mapping}

print(df['Emotion'].values)
print(le_mapping)

y = df['Emotion'].values

# Tokenize
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))


tokenized_text = [tokenize_reviews(review) for review in text]

text_with_len = [[review, y[i], len(review)]
                 for i, review in enumerate(tokenized_text)]

random.shuffle(text_with_len)
text_with_len.sort(key=lambda x: x[2])
sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in text_with_len]

processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))

BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,), ()))

print(next(iter(batched_dataset)))

TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

print(type(train_data))


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 6

DROPOUT_RATE = 0.2

NB_EPOCHS = 1

model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
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


