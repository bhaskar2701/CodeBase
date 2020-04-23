# Libraries
import os
os.chdir("D://trainings//NLP")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc; gc.enable()

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #tf.set_random_seed(seed_value)

#%%  Read data in Panda data frame for explorations
# It was gnerated in "2.2.nlp_intermediate_text_to_features.py"
train = pd.read_csv("./data/Reviews_5000_cleaned_countvec.csv")
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

#Drop response column as not required for clustering
train.drop(strResponse, axis=1, inplace=True)

train.head(2)

# Getting lists of IV
listAllPredictiveFeatures = train.columns

#%% Latent Dirichlet Allocation (LDA) : Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation

# LDA need count of topics beforehand. Use ML knowledge to determine
# let us use
num_clusters = 5; num_top_words = 10

# Create object
lda = LatentDirichletAllocation(n_components=num_clusters, max_iter=100,
                                learning_method='online', # if the data size is large, the online update will be much faster than the batch update.
                                learning_offset=50., # A (positive) parameter that downweights early iterations in online learning. It should be greater than 1.0
                                random_state=seed_value)

# fit (means train on data)
lda.fit(train)

# Top few words per topic
for topic_idx, topic in enumerate(lda.components_):
    message = "Topic #%d: " % topic_idx
    message += " ".join([train.columns[i] + '*' + str(round(topic[i], 1)) for i in topic.argsort()[:-num_top_words - 1:-1]])
    print(message)

#Topic #0: GREEN*2037.7 LOVE*1425.6 TEETH*1054.7 TREAT*1039.7 PRICE*718.8 GREAT*674.6 PRODUCT*573.8 BREATH*532.2 CLEAN*502.9 TIME*442.4
#Topic #1: COLONI*881.2 COOK*877.9 SOFT*760.1 LOVE*572.1 OATMEAL*504.4 TAST*481.6 GOOD*386.4 QUAKER*359.9 GREAT*305.5 BAKE*272.4
#Topic #2: TEETH*170.7 GREEN*156.5 EASI*93.6 RECOMMEND*91.8 ORDER*77.1 LOVE*67.5 TREAT*60.9 SIZE*60.5 HIGHLI*56.5 CLEAN*51.1
#Topic #3: GREEN*132.6 TREAT*89.9 BREATH*56.5 LOVE*53.0 CRAZI*44.9 YEAR*41.2 DONT*41.0 CHEW*36.6 KISS*36.6 PRODUCT*36.0
#Topic #4: GREEN*230.2 PRODUCT*103.1 TREAT*85.1 LOVE*67.1 DIGEST*62.4 REVIEW*58.7 DONT*54.4 FOOD*48.2 READ*46.8 REGULAR*46.2

del(train, strResponse, listAllPredictiveFeatures, num_clusters, lda, topic_idx, topic, message)

#%% Just FYI: GuidedLDA
#https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164