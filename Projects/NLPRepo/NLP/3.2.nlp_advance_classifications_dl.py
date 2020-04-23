# Libraries
import os
os.chdir("D://trainings//NLP")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc; gc.enable()

import tensorflow as tf
from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix
#from sklearn.metrics import cohen_kappa_score, confusion_matrix

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #tf.set_random_seed(seed_value)

#%% Multi class Dl with tf.keras MLP on TF-IDF data
#Read data in Panda data frame for explorations
# It was gnerated in "2.2.nlp_intermediate_text_to_features.py"
train = pd.read_csv("./data/Reviews_5000_cleaned_tfidf.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head(2)

# Drop 'TEXT' to save some memory
train.drop('TEXT', axis=1, inplace=True)

# Define response column
strResponse = 'SCORE'

# it may happen that one of the column name is 'score' hence rename response
if len(train[strResponse].shape) > 1:
    if train[strResponse].shape[1] > 1:
        strResponse = 'SCORE_RESPONSE'
        new_cols = list(train.columns)
        new_cols[0] = strResponse
        train.columns = new_cols
        del(new_cols)

# now convert the types
train[strResponse] = pd.to_numeric(train[strResponse], errors='coerce')
train.dtypes

# DL need 0 onwards
train[strResponse] = train[strResponse] -1

#Split the data into train and test
train, test = train_test_split(train, test_size=0.15, stratify = train[strResponse], random_state=seed_value)

# See if all score are present in both train and test
train.groupby(strResponse).size()
test.groupby(strResponse).size()

# Getting lists of IV
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(int(train.shape[1]/2), input_shape=(train[listAllPredictiveFeatures].shape[1],), activation=tf.nn.relu)) # , kernel_initializer = tf.random_normal_initializer
model.add(tf.keras.layers.Dense(int(train.shape[1]/4), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(int(train.shape[1]/8), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(int(train.shape[1]/16), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax)) # , kernel_initializer = tf.random_normal_initializer

#Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train it
model.fit(train[listAllPredictiveFeatures].values, train[strResponse].values, epochs=10,batch_size=16, workers = os.cpu_count(), use_multiprocessing = True)

# Evaluate on test data
model.evaluate(test[listAllPredictiveFeatures].values, test[strResponse].values)
#5000: loss value & metrics values: [0.59, 0.9]

#Making Predictions
predictions = model.predict(x=test[listAllPredictiveFeatures].values, verbose=1)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(test[strResponse].values, predictions_number)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#5000: Overall Accuracy is  0.9 , Kappa is  0.75

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

del(train, strResponse, test, listAllPredictiveFeatures, model, predictions, confusion_matrix, cms, df, predictions_number)

# Improvement: Will be discusssed later at the end of this file

#%% Multi class classification with tf.keras with inbuilt DL embedded layer
#Read data in Panda data frame for explorations
# It was gnerated in "2.2.nlp_intermediate_text_to_features.py"
train = pd.read_csv("./data/Reviews_5000_cleaned.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head(2)

# now convert the types
strResponse = 'SCORE'
train[strResponse] = pd.to_numeric(train[strResponse], errors='coerce')

# DL need 0 onwards
train[strResponse] = train[strResponse] -1

# Calculate few constants from train data
# Define dummy - just for some idea
max_number_words = 3000; EMBEDDING_DIM = 10; max_sentence_length = 0

# get all words and max length of any sentences
all_words = []
for sentence in train['TEXT']:
    words = sentence.split()
    all_words.extend(words)
    lw = len(words)
    #max_sentence_length = lw if lw > max_sentence_length else max_sentence_length
    if lw > max_sentence_length:
        max_sentence_length = lw

#a helper funtion to get unique words only
def get_unique_words(all_words):
    unique_words = []
    for word in all_words:
          if not word in unique_words:
              unique_words.append(word);

    return unique_words
# end of get_unique_words

# get unique set of words
all_unique_words = get_unique_words(all_words)

# count of all unique words
max_number_words = len(all_unique_words)

# cleaning
del(all_unique_words, all_words, words, lw)

# Encoding in numeric values
encoded_docs = [tf.keras.preprocessing.text.one_hot(sentence, max_number_words) for sentence in train['TEXT']]
encoded_docs[:2]

# pad documents to a max length of max_sentence_length words
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_sentence_length, padding='post')
padded_docs[:2]

df = pd.DataFrame(padded_docs)
listAllPredictiveFeatures = [''.join(['w',str(col)]) for col in df.columns]
df.columns = listAllPredictiveFeatures

# The padded data is new train data. Concatenate in original
train = pd.concat([train, df], axis = 1, join = "outer") # cbind

# Drop 'TEXT' to save some memory
train.drop('TEXT', axis=1, inplace=True)
del(df, padded_docs, encoded_docs)

#Split the data into train and test
train, test = train_test_split(train, test_size=0.15, stratify = train[strResponse], random_state=seed_value)

# See if all score are present in both train and test
train.groupby(strResponse).size()
test.groupby(strResponse).size()

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_number_words, EMBEDDING_DIM, input_length=max_sentence_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(int(EMBEDDING_DIM * max_sentence_length/2), activation=tf.nn.relu)) # input_shape=(train[listAllPredictiveFeatures].shape[1],), kernel_initializer = tf.random_normal_initializer
#model.add(tf.keras.layers.Dense(int(EMBEDDING_DIM * max_sentence_length/4), activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(int(EMBEDDING_DIM * max_sentence_length/8), activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(int(EMBEDDING_DIM * max_sentence_length/16), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax)) # , kernel_initializer = tf.random_normal_initializer

#Compile
#The string 'adam' is throwing warning "Converting sparse IndexedSlices to a dense Tensor of unknown shape."
#hence using adam object tf.train.AdamOptimizer()
#https://github.com/keras-team/keras/issues/4365#issuecomment-260482550
model.compile(optimizer= tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # binary_crossentropy
model.summary()

# Train it
model.fit(train[listAllPredictiveFeatures], train[strResponse].values, epochs=10,batch_size=16, workers = os.cpu_count(), use_multiprocessing = True)

# Evaluate on test data
model.evaluate(test[listAllPredictiveFeatures].values, test[strResponse].values)
#5000: loss value & metrics values: [.63, 0.87]

#Making Predictions
predictions = model.predict(x=test[listAllPredictiveFeatures].values, verbose=1)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(test[strResponse].values, predictions_number)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#5000-10: Overall Accuracy is  0.88 , Kappa is  0.68

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

del(train, strResponse, test, listAllPredictiveFeatures, model, predictions, confusion_matrix, cms, df, max_number_words, EMBEDDING_DIM, max_sentence_length, predictions_number, sentence)

# Improvement: Will be discusssed later at the end of this file
#%% Multi class classification with tf.keras with WordVector embedded layer transformed to average
#after taking average
from gensim.models import Word2Vec

train = pd.read_csv("./data/Reviews_5000_cleaned.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head(2)

# load wv model
model = Word2Vec.load('./model/wv_review_model_5000.bin')
print(model)

# Each row in data are consist of many words. Hence data is in shape (embedded
#dim (100) * number of words). Let get average of each row. After it each row
# will have embedded dim (100) data
dict_row_avg_embed = {}; i = 0
for sentence in train['TEXT']:
    average_vector = (np.mean(np.array([model.wv[word] for word in sentence.split()]), axis=0))
    dict_temp = { i : (average_vector) }; i = i+ 1
    dict_row_avg_embed.update(dict_temp)

# see the data shape
len(dict_row_avg_embed) # Total size
dict_row_avg_embed[0].shape # Dimension of each element

# Convert to DF for better processing
df = pd.DataFrame(dict_row_avg_embed)
df = df.transpose()
df.shape
df.head(2)

listAllPredictiveFeatures = [''.join(['w',str(col)]) for col in df.columns]
df.columns = listAllPredictiveFeatures

# The padded data is new train data. Concatenate in original
train = pd.concat([train, df], axis = 1, join = "outer") # cbind

# Drop 'TEXT' to save some memory
train.drop('TEXT', axis=1, inplace=True)
del(df, dict_row_avg_embed, i, dict_temp, average_vector, sentence)

# Define response column
strResponse = 'SCORE'

# DL need 0 onwards
train[strResponse] = train[strResponse] -1

#Split the data into train and test
train, test = train_test_split(train, test_size=0.15, stratify = train[strResponse], random_state=seed_value)

# See if all score are present in both train and test
train.groupby(strResponse).size()
test.groupby(strResponse).size()

# Convert to matrix for train and test data too
Y_train = tf.keras.utils.to_categorical(train[strResponse].values, num_classes=len(train[strResponse].unique()))
Y_train.shape
Y_test = tf.keras.utils.to_categorical(test[strResponse].values, num_classes=len(test[strResponse].unique()))
Y_test.shape

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(int(train.shape[1]/2), input_shape=(train[listAllPredictiveFeatures].shape[1],), activation=tf.nn.relu)) # , kernel_initializer = tf.random_normal_initializer
model.add(tf.keras.layers.Dense(int(train.shape[1]/4), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(int(train.shape[1]/8), activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(int(train.shape[1]/16), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax)) # , kernel_initializer = tf.random_normal_initializer

#Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # binary_crossentropy sparse_categorical_crossentropy
model.summary()

# Train it
model.fit(train[listAllPredictiveFeatures], Y_train, epochs=1000,batch_size=64, workers = os.cpu_count(), use_multiprocessing = True)

# Evaluate on test data
model.evaluate(test[listAllPredictiveFeatures].values, Y_test)
#1000: loss value & metrics values: [.73, 0.74]
#10000: loss value & metrics values: [.49, 0.79]

#Making Predictions
predictions = model.predict(x=test[listAllPredictiveFeatures].values, verbose=1)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(test[strResponse].values, predictions_number)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#1000: Overall Accuracy is  0.74 , Kappa is  0.11
#10000: Overall Accuracy is  0.79 , Kappa is  0.45

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

del(train, strResponse, test, listAllPredictiveFeatures, model, predictions, confusion_matrix, cms, df, predictions_number, Y_train, Y_test)

# Improvement: See the slide -> Hyper tunning, More data

#%% Multi class classification with tf.keras with customised WordVector embedded layer
from gensim.models import Word2Vec

train = pd.read_csv("./data/Reviews_5000_cleaned_tfidf.csv")
# Make column in lower case as word embedding is in lower case
train.columns = map(str.lower, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head(2)

# Define response column
strResponse = 'score'

# it may happen that one of the column name is 'score' hence rename response
if len(train[strResponse].shape) > 1:
    if train[strResponse].shape[1] > 1:
        strResponse = 'score_response'
        new_cols = list(train.columns)
        new_cols[0] = strResponse
        train.columns = new_cols
        del(new_cols)

# now convert the types
train[strResponse] = pd.to_numeric(train[strResponse], errors='coerce')
train.dtypes

# Drop 'TEXT' to save some memory and make sure train is having only word column
train.drop(['text'], axis=1, inplace=True)

#Split the data into train and test
train, test = train_test_split(train, test_size=0.15, stratify = train[strResponse], random_state=seed_value)

# Rest index as we will be using index to iterate in word embedding transformation
train.reset_index(drop=True, inplace =True)
test.reset_index(drop=True, inplace =True)

# DL need 0 onwards
actual_train = train[strResponse] -1
actual_test = test[strResponse] -1

# Drop response to save some memory and make sure data is having only word column
train.drop([strResponse], axis=1, inplace=True)
test.drop([strResponse], axis=1, inplace=True)

# load wv model
model = Word2Vec.load('./model/wv_review_model_5000.bin')
print(model)

# Word embedding dimension
word_embedding_dim = model.wv[train.columns[0]].size

# Explain data structure in XL.
# Each row in data are consist of many words (here in columns). Hence data is in shape
# (rows * number of words). Let us get in form of  (rows * (word_embedding_dim) * number of words)
def get_df_in_word_embedding_dim(train, word_embedding_dim, model):
    train_data = np.zeros((train.shape[0], word_embedding_dim, train.shape[1]), dtype = np.float)
    for row_num in range(train.shape[0]):
        df_word_embedding = pd.DataFrame(data = np.zeros((word_embedding_dim,train.shape[1])), columns=train.columns, dtype = np.float)
        for col_name in train.columns:
            if train.loc[row_num, col_name] > 0.0: # row_num = 2 ; col_name = 'abat'
                df_word_embedding[col_name] = np.ravel(model.wv[col_name])
        train_data[row_num] = df_word_embedding.values
        del(df_word_embedding)

    return(train_data)

# end of get_df_in_word_embedding_dim

# Call above function for transformation
train_data = get_df_in_word_embedding_dim(train, word_embedding_dim, model)
test_data = get_df_in_word_embedding_dim(test, word_embedding_dim, model)

# Now verify for 1 record
df = pd.DataFrame(train_data[0])
df.to_csv('./data/np.csv', index=False)

# clean
del(df, train, test); gc.collect()

# Convert to matrix for train and test data too
NUM_LABELS = len(actual_train.unique())
actual_train_he = tf.keras.utils.to_categorical(actual_train, num_classes= NUM_LABELS)
actual_train_he.shape # (2919, 5)
actual_test_he = tf.keras.utils.to_categorical(actual_test.values, num_classes=NUM_LABELS)
actual_test_he.shape # (516, 5)

#Just advice: Since above data prepration takes time and hence advise to save in actual project
np.save('./data/Reviews_5000_cleaned_word_embedding_train.npy', train_data)
np.save('./data/Reviews_5000_cleaned_word_embedding_train_actual.npy', actual_train_he)
np.save('./data/Reviews_5000_cleaned_word_embedding_test.npy', test_data)
np.save('./data/Reviews_5000_cleaned_word_embedding_test_actual.npy', actual_test_he)

#Load above data
train_data = np.load('./data/Reviews_5000_cleaned_word_embedding_train.npy')
actual_train_he = np.load('./data/Reviews_5000_cleaned_word_embedding_train_actual.npy')
test_data = np.load('./data/Reviews_5000_cleaned_word_embedding_test.npy')
actual_test_he = np.load('./data/Reviews_5000_cleaned_word_embedding_test_actual.npy')

#train_data = train_data[:,:, :,np.newaxis]
#test_data = test_data[:,:, :,np.newaxis]

# Because of memory issue, unable to proceed for model building using Conv2D

del(train_data, actual_train_he, actual_train, test_data, actual_test_he, actual_test)

#%% ppt: How to know models are good enough: Bias vs Variance
