#%% Import libraries
import os
import numpy as np
import pandas as pd

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# Working directory
os.chdir("D://trainings//NLP")

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#%% Text to Features
# Theory on ppt

#%% One Hot Encoding

# Create dummy data
train = pd.DataFrame([['I am learning NLP'], ['learning NLP with book and NLP online course']], columns = ['TEXT'])
train

# get all words by concatenating and spliting
all_words = train['TEXT'].str.cat(sep=' ').split()

#a helper funtion to get unique words only
def get_unique_words(all_words):
    unique_words = []
    for word in all_words:
          if not word in unique_words:
              unique_words.append(word);

    return unique_words
# end of get_unique_words

# Call above to get unique words
all_words = get_unique_words(all_words)

# A helper function create hot encoding by taking 0 and 1
def get_he(str_row_text, all_words):
    he = []
    for all_word in all_words:
        if all_word in str_row_text:
            he.extend([1])
        else:
            he.extend([0])

    return(he)
# end of get_he

#Call for each row
train_he = train['TEXT'].apply(lambda str_row_text: get_he(str_row_text, all_words))

# Convert to DF
train_he = pd.DataFrame(train_he.to_list(), columns=all_words)

# Concatenate in original
train = pd.concat([train, train_he], axis = 1, join = "outer") # cbind
train
# Observe: NLP occurued two time in line 2 and still have same weiatage like line 1.
# Drawback of one hot encoding: multiple times appearance has no extra weiatage
#A count vectorizer will solve that problem.

del(train, train_he, all_words)

#%% Count vectorizer on dummy data
from sklearn.feature_extraction.text import CountVectorizer

# Create dummy data
train = pd.DataFrame([['I am learning NLP'], ['learning NLP with book and NLP online course']], columns = ['TEXT'])
train

# get all words by concatenating and spliting
all_words = train['TEXT'].str.cat(sep=' ')

# create the object
count_vectorizer = CountVectorizer()

# fit it (means train on all words)
count_vectorizer.fit([all_words])

# Transform for the given data
text_transform = count_vectorizer.transform([train.loc[1,'TEXT']])
count_vectorizer.vocabulary_
text_transform.toarray()

# Looks good although combination of words is not captured

# create the object
count_vectorizer = CountVectorizer(ngram_range=(1, # Min 1 word
                                                2)) # Max 2 words

# fit it (means train on all words)
count_vectorizer.fit([all_words])

# Transform for the given data
text_transform = count_vectorizer.transform([train.loc[1,'TEXT']])
count_vectorizer.vocabulary_
text_transform.toarray()

#%% Count vectorizer on Review data
from sklearn.feature_extraction.text import CountVectorizer

# It was gnerated in "1.5.nlp_basic_string_cleaning.py"
train = pd.read_csv("./data/Reviews_5000_cleaned.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head()

# get all words by concatenating and spliting
all_words = train['TEXT'].str.cat(sep=' ')

# create the object
count_vectorizer = CountVectorizer()

# fit it (means train on all words)
count_vectorizer.fit([all_words])

# Transform for the given data
count_vectorizer.vocabulary_

# https://stackoverflow.com/questions/52141785/sort-dict-by-values-in-python-3-6
vocabulary_sorted = {k: v for k, v in sorted(count_vectorizer.vocabulary_.items(), key=lambda x: x[1])}

# Transform for the given data
text_transform = count_vectorizer.transform(train['TEXT'].tolist())
text_transform.shape
#text_transform.toarray()

# Convert to DF
text_transform = pd.DataFrame(text_transform.todense(), columns=vocabulary_sorted.keys()) #

# Concatenate in original
train = pd.concat([train, text_transform], axis = 1, join = "outer") # cbind

# Let us save in temp file to see XL
train.to_csv("./data/Reviews_5000_cleaned_countvec.csv",index = False)

del(train, text_transform, vocabulary_sorted, count_vectorizer, all_words)

#%% TF-IDF
#TF-IDF stands for "Term Frequency, Inverse Document Frequency". It is a way to score the importance of
#words (or "terms") in a document based on how frequently they appear across multiple documents.
#If a word appears frequently in a document, it's important. Give the word a high score.
#But if a word appears in many documents, it's not a unique identifier. Give the word a low score.
#Therefore, common words like "the" and "for", which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.

# TF * IDF = [ (Number of times term t appears in a document) / (Total number of terms in the document) ]
#            * log10(Total number of documents / Number of documents with term t in it)
# You may also read document as row of each vector

#Import library
from sklearn.feature_extraction.text import TfidfVectorizer

# Create dummy data
train = pd.DataFrame([['I am learning NLP'], ['learning NLP with book and NLP online course']], columns = ['TEXT'])
train

# Make lower case
train['TEXT'] = train['TEXT'].str.lower()

# create the object
tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 5000)

# fit it (means train on all sentences)
tfidf_vectorizer.fit(train['TEXT'].tolist())

#Summarize
tfidf_vectorizer.vocabulary_
tfidf_vectorizer.idf_

# https://stackoverflow.com/questions/52141785/sort-dict-by-values-in-python-3-6
vocabulary_sorted = {k: v for k, v in sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda x: x[1])}

# Transform for the given data
text_transform = tfidf_vectorizer.transform(train['TEXT'].tolist())
text_transform.shape
text_transform.toarray()

# Convert to DF
text_transform = pd.DataFrame(text_transform.todense(), columns=vocabulary_sorted.keys()) #

# Concatenate in original
train = pd.concat([train, text_transform], axis = 1, join = "outer") # cbind
train

#%% Let us transform 'Reviews' data for future explorations

#Import library
from sklearn.feature_extraction.text import TfidfVectorizer

# It was gnerated in "1.5.nlp_basic_string_cleaning.py"
train = pd.read_csv("./data/Reviews_5000_cleaned.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head()

#Transform to TF-IDF vectors
# create the object
tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 5000)

# fit it (means train on all sentences)
tfidf_vectorizer.fit(train['TEXT'].tolist())

#Summarize
len(tfidf_vectorizer.vocabulary_)
tfidf_vectorizer.vocabulary_
tfidf_vectorizer.idf_

# https://stackoverflow.com/questions/52141785/sort-dict-by-values-in-python-3-6
vocabulary_sorted = {k: v for k, v in sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda x: x[1])}

# Transform for the given data
text_transform = tfidf_vectorizer.transform(train['TEXT'].tolist())
text_transform.shape
#text_transform.toarray()

# Convert to DF
text_transform = pd.DataFrame(text_transform.todense(), columns=vocabulary_sorted.keys()) #

# Concatenate in original
train = pd.concat([train, text_transform], axis = 1, join = "outer") # cbind

# Let us save in temp file to see XL
train.to_csv("./data/Reviews_5000_cleaned_tfidf.csv",index = False)
