# Libraries
import os
os.chdir("D://trainings//NLP")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc; gc.enable()

from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #tf.set_random_seed(seed_value)

#%%  Read data in Panda data frame for explorations
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

#Split the data into train and test
train, test = train_test_split(train, test_size=0.15, stratify = train[strResponse],random_state=seed_value)

# See if all score are present in both train and test
train.groupby(strResponse).size()
test.groupby(strResponse).size()

# Few common params
# Getting lists of IV
listAllPredictiveFeatures = np.setdiff1d(train.columns,strResponse)

#%% Random forest
from sklearn.ensemble import RandomForestClassifier

# Default training param for RF
param = {'n_estimators': 5000, # The number of trees in the forest.
         'min_samples_leaf': 5, # The minimum number of samples required to be at a leaf node (external node)
         'min_samples_split': 10, # The minimum number of samples required to split an internal node (will have further splits)
         'max_depth': None, 'bootstrap': True, 'max_features': "auto", # The number of features to consider when looking for the best split
          'verbose': True, 'random_state': seed_value, 'n_jobs' : 4} # , 'warm_start' : True

#Build model on training data
classifier = RandomForestClassifier(**param)
classifier = classifier.fit(train[listAllPredictiveFeatures],train[strResponse])

# Self predict
predictions = classifier.predict(train[listAllPredictiveFeatures])

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(train[strResponse].tolist(), predictions.tolist())
confusion_matrix

# Few stats
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))

# Predict on test data
predictions = classifier.predict(test[listAllPredictiveFeatures])
confusion_matrix = ConfusionMatrix(test[strResponse].tolist(), predictions.tolist())
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#5000: Overall Accuracy is  0.8 , Kappa is  0.24

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')] # 101 line

# How to predict probabilities of each class
pred_prob = classifier.predict_proba(test[listAllPredictiveFeatures])
pred_prob[:2]
pred_class_index = np.argmax(pred_prob, axis=1)

del(cms, confusion_matrix, predictions, classifier, param, pred_class_index, pred_prob, df )
#%% Naive Bayes
from sklearn import naive_bayes

#Build model on training data
classifier = naive_bayes.MultinomialNB(alpha=0.2)
classifier = classifier.fit(train[listAllPredictiveFeatures],train[strResponse])

# Self predict
predictions = classifier.predict(train[listAllPredictiveFeatures])

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(train[strResponse].tolist(), predictions.tolist())
confusion_matrix

# Few stats
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))

# Predict on test data
predictions = classifier.predict(test[listAllPredictiveFeatures])
confusion_matrix = ConfusionMatrix(test[strResponse].tolist(), predictions.tolist())
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#Overall Accuracy is  0.86 , Kappa is  0.58

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')] # 101 line

# How to predict probabilities of each class
pred_prob = classifier.predict_proba(test[listAllPredictiveFeatures])
pred_prob[:2]
pred_class_index = np.argmax(pred_prob, axis=1)

del(cms, confusion_matrix, predictions, classifier, pred_class_index, pred_prob, df )

#%% Xgboost
import xgboost as xgb

# for xgb label should start with 0

# Get data in form DMatrix as required by xgboost
d_train = xgb.DMatrix(train[listAllPredictiveFeatures], label=train[strResponse]-1)

# https://xgboost.readthedocs.io/en/latest/parameter.html
params = {'eval_metric' : 'merror', 'eta': 0.3, 'min_child_weight': 1,
          'colsample_bytree': 1, 'subsample': 1, 'max_depth': 6, 'nthread' : 4,
          'booster' : 'gbtree', 'objective' : "multi:softmax", 'num_class' : 5, 'random_state': seed_value} # multi:softprob

#Build model on training data
classifier = xgb.train(dtrain = d_train, params = params,num_boost_round=100)

#Self Prediction
predictions = classifier.predict(d_train)

# Compute confusion matrix
confusion_matrix = ConfusionMatrix((train[strResponse]-1).tolist(), predictions.tolist())
confusion_matrix

# Few stats
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))

# Predict on test data
d_test = xgb.DMatrix(test[listAllPredictiveFeatures])
predictions = classifier.predict(d_test)
confusion_matrix = ConfusionMatrix((test[strResponse]-1).tolist(), predictions)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#Overall Accuracy is  0.89 , Kappa is  0.71

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

del(cms, confusion_matrix, predictions, classifier, df )

#RF: Overall Accuracy is  0.8 , Kappa is  0.24
#NB: Overall Accuracy is  0.86 , Kappa is  0.58
#XG: Overall Accuracy is  0.89 , Kappa is  0.71

# Improvement
# ngram_range=(1,3) in TfidfVectorizer
# More data
