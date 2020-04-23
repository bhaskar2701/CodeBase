# Theory on ppt
# Libraries
import os
os.chdir("D://trainings//NLP")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc; gc.enable()

from pandas_ml import ConfusionMatrix

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #tf.set_random_seed(seed_value)

#%% Customised Sentiment
# 3.1.nlp_advance_classifications_ml
# 3.2.nlp_advance_classifications_dl

#%% TextBlob

#%%VADER Sentiment Analysis:VADER(Valence Aware Dictionary ands Entiment Reasoner)is
#a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments
#expressed in socialmedia.

#%%  VADER Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import cohen_kappa_score, confusion_matrix

#Read data in Panda data frame for explorations
# It was gnerated in "3.nlp_basic_string_cleaning.py"
train = pd.read_csv("./data/Reviews_5000_cleaned.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head(2)

# Define response column
strResponse = 'SCORE'

# Let us create group - negative for score 1 & 2 and positive for 4 & 5. Drop 3 - neutral
train = train[train[strResponse] != 3]

mapping = {1 : 'Negative',2 : 'Negative',4 : 'Positive', 5 : 'Positive'}
train[strResponse] = train[strResponse].replace(mapping)

#CW: See word cloud of all positive reviews texts
#CW: See word cloud of all negative reviews texts

# create object
vader_sentiment_analyzer = SentimentIntensityAnalyzer()

# Generating sentiment for all the sentence
list_sentiments=[]
for row_text in train['TEXT']:
    list_sentiments.append(vader_sentiment_analyzer.polarity_scores(row_text))

# Creating new dataframe with sentiments
df_sentiments=pd.DataFrame(list_sentiments)
df_sentiments.head(5)

# Get consolidated 'Negative' or 'Positive' and Append to main data for further analysis
train['SENTIMENT'] = np.where(df_sentiments['compound'] >= 0 , 'Positive', 'Negative')

train.head()

# Confusion matrix
TN, FP, FN, TP = confusion_matrix(train[strResponse].tolist(), train['SENTIMENT'].tolist()).ravel()

# Extract metrics
accuracy = round(float(TP + TN)/train.shape[0], 2)
kappa = round(cohen_kappa_score(train[strResponse].tolist(), train['SENTIMENT'].tolist()),2)
specificity = round(float(TN)/float(TN+FP) , 2); sensitivity = round(float(TP)/float(TP+FN) , 2)

print("Overall Accuracy is ", accuracy,", Kappa is ", kappa)
#Overall Accuracy is  0.93 , Kappa is  0.33
print("Specificity is ", specificity,", Sensitivity is ", sensitivity)
#Specificity is  0.4 , Sensitivity is  0.96

del(train, strResponse, mapping, vader_sentiment_analyzer, list_sentiments, df_sentiments, TN, FP, FN, TP, accuracy, kappa, specificity, sensitivity)

#%% CW: Sentiment as classification : Data prepration is shown here

#Read data in Panda data frame for explorations
# It was gnerated in "3.nlp_basic_string_cleaning.py"
train = pd.read_csv("./data/Reviews_5000_cleaned_tfidf.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head(2)

# Define response column
strResponse = 'SCORE'

# Let us create group - negative for score 1 & 2 and positive for 4 & 5. Drop 3 - neutral
train = train[train[strResponse] != 3]

# Replace in 0 (-ve) and 1 (+ve) format
mapping = {1 : 0, 2 : 0}; train[strResponse] = train[strResponse].replace(mapping)
mapping = {4 : 1, 5 : 1}; train[strResponse] = train[strResponse].replace(mapping)

train.head(2)
