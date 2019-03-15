#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:42:33 2019

@author: kristen
"""


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




# Import BERT vectors and data
#BERT embeddings downloaded from
#https://github.com/Kyubyong/bert-token-embeddings
EMBEDDING_FILE='bert-base-uncased.30522.768d.vec'
TRAIN_DATA_FILE='train.csv'
MODEL_NAME = 'bert_cnn'

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
    
    
# CNN model
model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(128, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(dropout_rate))
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

file = open('/home/ec2-user/bert_cnn/bert_cnn_results.txt','w') 

file.write(final_auc_string)
file.write(class_auc_string)

file.close()














