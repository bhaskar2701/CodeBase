# Libraries
import os
os.chdir("D://trainings//NLP")

import numpy as np
import pandas as pd
import gc; gc.enable()

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #tf.set_random_seed(seed_value)

#%% NER is covered in section "Part-of-speech tagging" in file '1.4.nlp_basic_nltk_operations.py'

#Already covered
#Machine Translation
#Text Summarization
#Speech to Text

#%% Computational Linguistics can be performed using NLTK and Stanford-CoreNLP too

#%%Using spacy just for simple graphics to display the result
import spacy
from spacy import displacy

# get object
# python -m spacy download en  # install in admin mode and restart Spyder
nlp = spacy.load("en")

# sample text. Source: https://nlp.stanford.edu/software/dependencies_manual.pdf
spacy_tokens = nlp(u'Bell, based in Los Angeles, makes and distributes electronic, computer and building products')

# Print few attributes
for token in spacy_tokens:
    print('%s %s %s %s' %(str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_)))
    #token.tag_, token.shape_, token.is_alpha, token.is_stop

# After running follwoing command view the tree at http://localhost:5000/
displacy.serve(spacy_tokens,
                page = False, # No full HTML page
                port=5000, host='localhost',
                style='dep') # Visualisation style, 'dep' or 'ent'

