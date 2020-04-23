# Theory on ppt

# Libraries
import os
os.chdir("D://trainings//NLP")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc; gc.enable()

#Gensim Python Library: Gensim is an open source Python library for natural language
#processing with a focus on topic modeling. Gensim was developed and is maintained by
#the Czech natural language processing researcher Radim Řehůřek and his company RaRe Technologies.
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

import tensorflow as tf

from sklearn.decomposition import PCA

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); tf.set_random_seed(seed_value)

if not os.path.exists("model"):
    os.makedirs("model")

#%%Load Stanford’s GloVe Embedding
#The first step is to convert the GloVe file format to the word2vec file format. The only
#difference is the addition of a small header line. This can be done by calling the
#glove2word2vec() function. For example:

#You can download the smallest GloVe pre-trained model from the GloVe website. It an 822
#Megabyte zip file with 4 different models (50, 100, 200 and 300-dimensional vectors) trained
# on Wikipedia data with 6 billion tokens and a 400,000 word vocabulary.
#The direct download link is here: http://nlp.stanford.edu/data/glove.6B.zip

#Working with the 300-dimensional version of the model, we can convert the file to
#word2vec format as follows:
glove_input_file = './model/glove.6B/glove.6B.300d.txt'
word2vec_output_file = './model/glove.6B/glove.6B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
#You now have a copy of the GloVe model in word2vec format with the filename glove.6B.300d.txt.word2vec.

#Now we can load it and perform the same (king – man) + woman = ? test as in the previous
#section. The complete code listing is provided below. Note that the converted file is ASCII
#format, not binary, so we set binary=False when loading.

# load the Stanford GloVe model. It will take 5 min as file is 3.5GB (3 million words * 300 features * 4bytes/feature = 3.5GB)
word2vec_output_file = './model/glove.6B/glove.6B.300d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# to verify the presence of words in training data
if 'king' in model.vocab:
    print('king is present and represented by')
    print(model['king'])
else:
    print('king is not present')

# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
result # [('queen', 0.7698541283607483)]

model.most_similar(positive=['paris', 'italy' ], negative=['france'], topn=1)
model.most_similar(positive=['france', 'rome' ], negative=['paris'], topn=1)
model.most_similar(positive=['paris', 'india' ], negative=['france'], topn=1)
model.most_similar(positive=['apple', 'juice' ], topn=1)

# Can you guess the output
model.most_similar(positive=['apple'] , negative= ['juice' ], topn=1)

model.most_similar('apple')

model.similarity('woman','man')
model.similarity('women','men')
model.similarity('he', 'she')
model.similarity('laptop', 'desktop')

model.doesnt_match('breakfast cereal dinner lunch'.split())

# Clean up
os.remove(word2vec_output_file); del(model, result, word2vec_output_file, glove_input_file); gc.collect()

#%% CW: Load Google’s Word2Vec Embedding
#A pre-trained model is nothing more than a file containing tokens and their associated
#word vectors. The pre-trained Google word2vec model was trained on Google news data
#(about 100 billion words); it contains 3 million words and phrases and was fit using
#300-dimensional word vectors.

#It is a 1.53 Gigabytes file. You can download it from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
#
#Unzipped, the binary file (GoogleNews-vectors-negative300.bin) is 3.4 Gigabytes.
#
#The Gensim library provides tools to load this file. Specifically, you can call the
#KeyedVectors.load_word2vec_format() function to load this model into memory, for example:

# Load the file. It will take 1-2 min as file is 3.5GB
filename = './model/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

#Another interesting thing that you can do is do a little linear algebra arithmetic with words.
#queen = (king - man) + woman

#Gensim provides an interface for performing these types of operations in the most_similar()
#function on the trained or loaded model.
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
result #[('queen', 0.7118192315101624)]

model.most_similar(positive=['paris', 'italy' ], negative=['france'], topn=1)
# [('lohan', 0.5069674849510193)]
model.most_similar(positive=['paris', 'india' ], negative=['france'], topn=1)
#[('chennai', 0.5442506074905396)]
model.most_similar(positive=['apple', 'juice' ], topn=1)
#[('fruit', 0.6929349899291992)]

# Can you guess the output
model.most_similar(positive=['apple'] , negative= ['juice' ], topn=1)
#[('conifer', 0.36092501878738403)]

del(model); gc.collect()

#%% Basic: Word Embeddings by Word2Vec

#Develop Word2Vec Embedding
#It look at a window of words for each target word to provide context and in turn meaning
#for words. The approach was developed by Tomas Mikolov, formerly at Google and currently
#at Facebook.

# Few important parameters
#size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense
#vector to represent each token (word).
#window: (default 5) The maximum distance between a target word and words around the target word.
#min_count: (default 5) The minimum count of words to consider when training the model; words
# with an occurrence less than this count will be ignored.
#workers: (default 3) The number of threads to use while training.
#sg: (default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).

# Create dummy data
train = pd.DataFrame([['I am learning NLP'], ['learning NLP with book and NLP online course'],['Using text to feature to create word embedding for NLP'], ['It used deep learning to build NLP model']], columns = ['TEXT'])

# define training data in appropriate format
all_words= [sentence.lower().split() for sentence in train['TEXT']]

# train model
model = Word2Vec(all_words, min_count=1)

# summarize the loaded model
print(model)
# For alpha see explanation at https://stackoverflow.com/questions/53815402/value-of-alpha-in-gensim-word-embedding-word2vec-and-fasttext-models

# summarize vocabulary
print(list(model.wv.vocab)) # can reverify with np.unique(np.hstack(sentences))

# access vector for one word
print(model.wv['nlp'])

#Visualize Word Embedding
# Now let us view in 2d using dimension reduction with PCA
model_pca = PCA(n_components=2)
transform_data = model_pca.fit_transform(model.wv[model.wv.vocab])

# create a scatter plot of the projection
plt.scatter(transform_data[:, 0], transform_data[:, 1])
for i, word in enumerate(list(model.wv.vocab)):
    plt.annotate(word, xy=(transform_data[i, 0], transform_data[i, 1]))
plt.show()

#It is hard to pull much meaning out of the graph given such a tiny corpus was used to fit
#the model.

del(train, all_words, model, model_pca, transform_data)

#%% Review data: Word Embeddings by Word2Vec

# Read clean data
train = pd.read_csv("./data/Reviews_5000_cleaned.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# See the data
train.head()

# define training data in appropriate format
all_words= [sentence.lower().split() for sentence in train['TEXT']]

# train model
model = Word2Vec(all_words, size=100, min_count=1)

# summarize the loaded model
print(model)

# summarize vocabulary
print(list(model.wv.vocab)) # can reverify with np.unique(np.hstack(sentences))

# access vector for one word
print(model.wv['good'])

# save model in both binary and text format
model.save('./model/wv_review_model_5000.bin')
model.wv.save_word2vec_format('./model/wv_review_model_5000.txt', binary=False)

# load model
new_model = Word2Vec.load('./model/wv_review_model_5000.bin')
print(new_model)

#Visualize Word Embedding
# Now let us view in 2d using dimension reduction with PCA
model_pca = PCA(n_components=2)
transform_data = model_pca.fit_transform(model.wv[model.wv.vocab])

# create a scatter plot of the projection
plt.scatter(transform_data[:, 0], transform_data[:, 1])
for i, word in enumerate(list(model.wv.vocab)):
    plt.annotate(word, xy=(transform_data[i, 0], transform_data[i, 1]))
plt.show()

#It is hard to pull much meaning for so many words. Few selective words can be tried

del(train, all_words, model, new_model, model_pca, transform_data)

#%% CW: Explore FastText (An NLP library by Facebook)
#http://feedproxy.google.com/~r/AnalyticsVidhya/~3/r-TzzESKAbQ/?utm_source=feedburner&utm_medium=email

#%% Embeddings from Language Models (ELMo)
#https://allennlp.org/elmo
#https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
