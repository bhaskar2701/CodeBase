# Search engine - Also known as "Semantic search"
# Libraries
import os
os.chdir("D://trainings//NLP")

import numpy as np
import pandas as pd
import gc; gc.enable()

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #tf.set_random_seed(seed_value)

#%%  Read data in Panda data frame for explorations
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
#dim (100) * number fo words). Let get average of each row. After it each row
# will have embedded dim (100) data
dict_row_avg_embed = {}; i = 0
for sentence in train['TEXT']:
    average_vector = (np.mean(np.array([model.wv[word] for word in sentence.split()]), axis=0))
    dict_temp = { i : (average_vector) }; i = i+ 1
    dict_row_avg_embed.update(dict_temp)

# see the data shape
len(dict_row_avg_embed) # Total size
dict_row_avg_embed[0].shape # Dimension of each element

# Rank all the documents based on the similarity to get relevant docs
def compare_and_rank_documents(query): # query = 'tea'
    query_words = (np.mean(np.array([model.wv[word] for word in query.split() if word in model.wv.vocab]), axis=0))
    rank = []
    if query_words.size > 1:
        for k,v in dict_row_avg_embed.items():
            cos_sim = cosine_similarity(np.reshape(query_words, (1, -1)), np.reshape(v, (1, -1)))
            rank.append((k, cos_sim))
        rank = sorted(rank,key=lambda t: t[1], reverse=True)
    return(rank)
# end of def compare_and_rank_documents

top_ranked_records = compare_and_rank_documents("tea good") # tea good  nlp
#model.wv["nlp"]
if len(top_ranked_records) > 0:
    top_ranked_records[:2]
    train.loc[top_ranked_records[0][0],'TEXT'] # 'ahead tea expel looseleaf tea with hot make ice tea great flavor linger aftertast'
    train.loc[top_ranked_records[1][0],'TEXT'] # 'kind tea good last smell wear tea feel chemical'
