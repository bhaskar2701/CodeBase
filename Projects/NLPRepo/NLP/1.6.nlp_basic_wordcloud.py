#%% Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Working directory
os.chdir("D:/trainings/NLP")

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)


#%% Word Cloud
from wordcloud import  WordCloud

# Read afresh or continue from above
train = pd.read_csv("./data/Reviews_5000_cleaned.csv")

#Get the whole text
text = train['TEXT'].str.cat(sep=' ')
text[:1000]

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image using the matplotlib
plt.imshow(wordcloud, interpolation='bilinear'); plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
#background_color='white', stopwords= set(STOPWORDS), max_words=200, max_font_size=40,  scale=3, random_state=1
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off"); plt.show()
