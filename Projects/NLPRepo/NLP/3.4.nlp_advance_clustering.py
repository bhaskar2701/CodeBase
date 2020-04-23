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
# It was gnerated in "7.nlp_intermediate_text_to_features.py"
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

#Drop response column as not required for clustering
train.drop(strResponse, axis=1, inplace=True)

train.head(2)

# Getting lists of IV
listAllPredictiveFeatures = train.columns

#%% Clustering Using KMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# KMeans need count of cluster beforehand. Use ML knowledge to determine
# let us use
num_clusters = 5

# Create object
km = KMeans(n_clusters=num_clusters)

# fir (means train on data)
km.fit(train)

# Get cluster id as one column
train['CLUSTER_ID'] = km.labels_

# See the count
train['CLUSTER_ID'].value_counts()

# Top few words per cluster
for i in range(num_clusters): # i = 0
    print("Cluster %d words:" % i, end='')
    # get temp Df for each cluster
    df = train[train['CLUSTER_ID'] == i].copy()
    #Drop cluster id column as not required
    df.drop('CLUSTER_ID', axis=1, inplace=True)
    # Sum each row so that value (TF-IDF) for each word is obtained
    sum_each_col = df.sum(axis = 0)
    # get 5 top word with max value of TF-IDF
    sum_each_col.sort_values(ascending=False, inplace=True)
    print(sum_each_col[:5].index) # np.array( sum_each_col[:5].index)
    del(df, sum_each_col)
# end of for i in range(num_clusters)

#Cluster 0 words:Index(['COLONI', 'SOFT', 'TAST', 'OATMEAL', 'LOVE'], dtype='object')
#Cluster 1 words:Index(['PRICE', 'GREEN', 'STORE', 'GREAT', 'LOVE'], dtype='object')
#Cluster 2 words:Index(['TEETH', 'CLEAN', 'GREEN', 'LOVE', 'GREAT'], dtype='object')
#Cluster 3 words:Index(['COOK', 'SOFT', 'OATMEAL', 'INFLUENST', 'TAST'], dtype='object')
#Cluster 4 words:Index(['GREEN', 'TREAT', 'LOVE', 'BREATH', 'PRODUCT'], dtype='object')

# Now let us view in 2d using dimension reduction with PCA
model_pca = PCA(n_components=2)
transform_data = pd.DataFrame(model_pca.fit_transform(train[listAllPredictiveFeatures]))
transform_data.columns = ['PC1', 'PC2']
transform_data['CLUSTER_ID'] = km.labels_

#set up colors per clusters using a dict
cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black', 4: 'yellow'}
transform_data['COLOR'] = transform_data['CLUSTER_ID'].replace(cluster_colors)
transform_data.head(2)

# Create title
title = [str(k) + ': ' + str(v) for k,v in cluster_colors.items()]

# create a scatter plot of the projection
# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
plt.scatter(x = transform_data['PC1'], y = transform_data['PC2'], c = transform_data['COLOR'])
plt.title(title)
ax.set_aspect('auto')
ax.tick_params(axis= 'x', which='both', bottom=False, top=False, labelbottom=False)
ax.tick_params(axis= 'y', which='both', bottom=False, top=False, labelbottom=False)
plt.show()

# CW: reduce count and practice again

del(train, strResponse, listAllPredictiveFeatures, num_clusters, km, model_pca, transform_data, cluster_colors, title, fig, ax)

#CW: Find out distance using cosine and calculate cluster (Hiarchical)