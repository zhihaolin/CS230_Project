#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:16:34 2019

@author: kristen
"""

import gensim.models.keyedvectors as word2vec
import numpy as np, pandas as pd, datetime
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.models import Sequential
from keras.callbacks import CSVLogger
import keras.backend as K


# Import word2vec vectors and data
EMBEDDING_FILE='GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE='train.csv'
MODEL_NAME = 'w2v_cnn'

# set hyperparameters
embed_size = 300    # Word vector dimensionality                      
maxlen = 100 
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
epoch = 30

# read the data and embedding file, replace missing values
train = pd.read_csv(TRAIN_DATA_FILE)
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
word2vecDict = word2vec.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True, limit=500000)


# Turn each comment into a list of word indexes of equal length (with truncation or padding as needed).
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

# Split data into train and test set
X_train, X_val, y_train, y_val = train_test_split(X_t, y, train_size=0.9, random_state=233)

# embedding
embeddings_index = dict()
for word in word2vecDict.wv.vocab:
    embeddings_index[word] = word2vecDict.word_vec(word)
print('Loaded %s word vectors.' % len(embeddings_index))

all_embs = np.stack(list(embeddings_index.values()))
emb_mean,emb_std = all_embs.mean(), all_embs.std()
nb_words = len(tokenizer.word_index)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    
# CNN model
model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(128, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='sigmoid'))  #multi-label (k-hot encoding)

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

# Fit the model and predict 
model.fit(X_train, y_train, batch_size=32, epochs=epoch, validation_data=(X_val, y_val), callbacks=[csv_logger])


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

file = open('/home/ec2-user/w2v_cnn/w2v_cnn_results.txt','w') 

file.write(final_auc_string)
file.write(class_auc_string)

file.close()