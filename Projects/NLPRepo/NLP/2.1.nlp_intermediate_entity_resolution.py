#%% Import libraries
import os
import numpy as np
import pandas as pd

import recordlinkage

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# Working directory
os.chdir("D:/trainings/NLP")

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#%%  Entity resolution /Deduplication from single table
#Data sets are at link https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html
from recordlinkage.datasets import load_febrl1
#This data set contains 1000 records (500 original and 500 duplicates, with exactly
#one duplicate per original record.

#load data
train = load_febrl1()
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# Make list of data types and convert to approprite data types
list_text_data = ['GIVEN_NAME','SURNAME','STREET_NUMBER','ADDRESS_1','ADDRESS_2','SUBURB','STATE']
list_int_data = ['POSTCODE', 'SOC_SEC_ID']
train[list_text_data] = train[list_text_data].apply(lambda x: x.astype(str))
train[list_int_data] = train[list_int_data].apply(lambda x: x.astype(int))

# View summary
train.describe(include = 'all')

# See the data
train.head()

# Save data to view in XL
train.to_csv('./output/febrl1.csv', index = True)

# Discuss the simple search and drawback
indexer = recordlinkage.Index()
indexer.full()
candidate_links = indexer.index(train)
print (len(candidate_links ))
# (1000*1000-1000)/2 = 499500

# Discuss 'Blocking'

#Create index on block column
indexer = recordlinkage.index.Block(left_on='GIVEN_NAME') # Note: For huge number of records use blocking on multiple variables
candidate_links  = indexer.index(train)
print (len(candidate_links ))

# let us match and score
compare_rl = recordlinkage.Compare() # time consuming

# Parameter definition for matching
compare_rl.string('GIVEN_NAME', 'GIVEN_NAME',method='jarowinkler', label='GIVEN_NAME')
#Options are [‘jaro’, ‘jarowinkler’, ‘levenshtein’, ‘damerau_levenshtein’, ‘qgram’, ‘cosine’, ‘smith_waterman’, ‘lcs’]. Default: ‘levenshtein’

compare_rl.string('SURNAME', 'SURNAME', method='jarowinkler', threshold=0.85, # All approximate string comparisons higher or equal than this threshold are 1. Otherwise 0
                  label='SURNAME')
compare_rl.string('ADDRESS_1', 'ADDRESS_1',method='jarowinkler', threshold=0.6,label='ADDRESS_1')
compare_rl.exact('DATE_OF_BIRTH', 'DATE_OF_BIRTH', label='DATE_OF_BIRTH')
compare_rl.exact('SUBURB', 'SUBURB', label='SUBURB')
compare_rl.exact('STATE', 'STATE', label='STATE')

# Compute
df_comparison_results = compare_rl.compute(candidate_links , train)

# See the outcome
df_comparison_results.head()
df_comparison_results[df_comparison_results.sum(axis=1) > 3].head()

# Let us use unsupervised technique on all features except on which blocking is done

#refined data
list_features = ['SUBURB','STATE','SURNAME','DATE_OF_BIRTH','ADDRESS_1']
df_comparison_results = df_comparison_results[list_features]
df_comparison_results[list_features] = df_comparison_results[list_features].apply(lambda x: x.astype(int))
df_comparison_results.head()

# Build model object
classifier = recordlinkage.ECMClassifier()

#train
classifier.fit(df_comparison_results)

#Predict
pred = classifier.predict(df_comparison_results)

# Convert to Df for readability
df = pd.DataFrame([pred]).transpose()
df.head()

del(train, list_text_data, list_int_data, indexer, candidate_links , compare_rl, df_comparison_results, list_features, classifier, pred, df)
#%% Entity resolution /Deduplication from two table
# https://recordlinkage.readthedocs.io/en/latest/notebooks/link_two_dataframes.html
from recordlinkage.datasets import load_febrl4

#load data
train_one, train_two = load_febrl4()
train_one.columns = map(str.upper, train_one.columns)
train_two.columns = map(str.upper, train_two.columns)

# First view
train_one.shape, train_two.shape
train_one.dtypes
train_two.dtypes
train_one.info()
train_two.info()

# Make list of data types and convert to approprite data types
list_text_data = ['GIVEN_NAME','SURNAME','STREET_NUMBER','ADDRESS_1','ADDRESS_2','SUBURB','STATE']
list_int_data = ['POSTCODE', 'SOC_SEC_ID']
train_one[list_text_data] = train_one[list_text_data].apply(lambda x: x.astype(str))
train_one[list_int_data] = train_one[list_int_data].apply(lambda x: x.astype(int))
train_two[list_text_data] = train_two[list_text_data].apply(lambda x: x.astype(str))
train_two[list_int_data] = train_two[list_int_data].apply(lambda x: x.astype(int))

# View summary
train_one.describe(include = 'all')
train_two.describe(include = 'all')

# See the data
train_one.head()
train_two.head()

# Save data to view in XL
train_one.to_csv('./output/febrl4_one.csv', index = True)
train_two.to_csv('./output/febrl4_two.csv', index = True)

#Create index on block column
indexer = recordlinkage.index.Block(left_on='GIVEN_NAME',right_on='GIVEN_NAME')
candidate_links = indexer.index(train_one, train_two)
print (len(candidate_links ))

# let us match and score
compare_rl = recordlinkage.Compare() # time consuming

# Parameter definition for matching
compare_rl.string('GIVEN_NAME', 'GIVEN_NAME',method='jarowinkler', label='GIVEN_NAME')
#Options are [‘jaro’, ‘jarowinkler’, ‘levenshtein’, ‘damerau_levenshtein’, ‘qgram’, ‘cosine’, ‘smith_waterman’, ‘lcs’]. Default: ‘levenshtein’

compare_rl.string('SURNAME', 'SURNAME', method='jarowinkler', threshold=0.85, # All approximate string comparisons higher or equal than this threshold are 1. Otherwise 0
                  label='SURNAME')
compare_rl.string('ADDRESS_1', 'ADDRESS_1',method='jarowinkler', threshold=0.6,label='ADDRESS_1')
compare_rl.exact('DATE_OF_BIRTH', 'DATE_OF_BIRTH', label='DATE_OF_BIRTH')
compare_rl.exact('SUBURB', 'SUBURB', label='SUBURB')
compare_rl.exact('STATE', 'STATE', label='STATE')

# Compute. May take some time for huge data
df_comparison_results = compare_rl.compute(candidate_links , train_one, train_two)

# See the outcome
df_comparison_results.head()
df_comparison_results[df_comparison_results.sum(axis=1) > 3].head()

# Let us use unsupervised technique on all features except on which blocking is done

#refined data
list_features = ['SUBURB','STATE','SURNAME','DATE_OF_BIRTH','ADDRESS_1']
df_comparison_results = df_comparison_results[list_features]
df_comparison_results[list_features] = df_comparison_results[list_features].apply(lambda x: x.astype(int))

# Build model object
classifier = recordlinkage.ECMClassifier()

#train
classifier.fit(df_comparison_results)

#Predict
pred = classifier.predict(df_comparison_results)

# Convert to Df for readability
df = pd.DataFrame([pred]).transpose()
df.head()

del(train_one, train_two, list_text_data, list_int_data, indexer, candidate_links , compare_rl, df_comparison_results, list_features, classifier, pred, df)
#%% Libraries -> dedupe : One of the famous library in CLI mode
#pip install unidecode
#pip install future
#pip install dedupe
