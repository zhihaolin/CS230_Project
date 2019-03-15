#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:50:57 2019

@author: kristen
"""

import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras import initializers, regularizers, constraints, optimizers
from keras.models import Model
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional
from keras.callbacks import CSVLogger

# Import BERT vectors and data
#BERT embeddings downloaded from
#https://github.com/Kyubyong/bert-token-embeddings
EMBEDDING_FILE='bert-base-uncased.30522.768d.vec'
TRAIN_DATA_FILE='train.csv'
MODEL_NAME = 'bert_att'

# Set hyperparameters
embed_size = 300 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use
dropout_rate = 0.5
epoch = 30

# Read in data and replace missing values:
train = pd.read_csv(TRAIN_DATA_FILE)
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

# Turn each comment into a list of word indexes of equal length (with truncation or padding \
# as needed).
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

# Split train and test set
X_train, X_val, y_train, y_val = train_test_split(X_t, y, train_size=0.9, random_state=233)

# Read the bert embeddings (space delimited strings) into a dictionary from word->vector.
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

# Create embedding matrix with Bert embeddings
# with random initialization for words that aren't in GloVe. 
# Use same mean and stdev of embeddings the GloVe has when generating the random init.
all_embs = list(embeddings_index.values())
all_embs = np.stack(all_embs[1:])
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector[0:embed_size]


# LSTM model
# Attention layer
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

inp = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.25,recurrent_dropout=0.25))(x)
x = Attention(maxlen)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

def AUC(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', AUC])
model.summary()

# logger
now = datetime.datetime.now()
log_filename = str(MODEL_NAME)+now.strftime("%Y-%m-%d-%H%M")+str('.csv')
csv_logger = CSVLogger(log_filename, append=True, separator='|')

# fit the model, make prediction
model.fit(X_train, y_train, batch_size=32, epochs=epoch, validation_data=(X_val, y_val), callbacks=[csv_logger])

y_pred = model.predict(X_val)

# total AUC
y_pred = model.predict(X_val)
y_true = y_val
final_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
print(final_auc)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
class_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    class_auc[i] = auc(fpr[i], tpr[i])
print(class_auc)

final_auc_string = str(final_auc)
class_auc_string = str(class_auc)

file = open('/home/ec2-user/bert_att/bert_att_results.txt','w') 

file.write(final_auc_string)
file.write(class_auc_string)

file.close()


