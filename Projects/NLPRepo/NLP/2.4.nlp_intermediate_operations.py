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
os.chdir("D:/trainings/NLP")

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#%% Word Sense Disambiguation
from pywsd.lesk import simple_lesk

sent = 'I went to the bank to deposit money.'
answer = simple_lesk(sent,'bank')
answer, answer.definition() # See the meaning in SYsnet at bottom of this section

sent = 'I went to the bank to catch ship.'
answer = simple_lesk(sent,'bank')
answer, answer.definition()

from nltk.corpus import wordnet as wn
for ss in wn.synsets('bank'):
    print(ss, ss.definition())

#%% Speech Recognition using Microphone
# Theory on ppt
# https://pypi.org/project/SpeechRecognition/

import speech_recognition as sr

# Create recognizer object
recognizer=sr.Recognizer()

# Take microphone as source and listen
with sr.Microphone() as source:
    print("Listening start")
    # understands your voice in noisy room too
    recognizer.adjust_for_ambient_noise(source)
#    It wait till the audio has an energy above recognizer_instance.energy_threshold
#    (the user has started speaking), and then recording until it encounters
#    recognizer_instance.pause_threshold seconds of non-speaking or there is no more
#    audio input. The ending silence is not included.

    audio = recognizer.listen(source) # , timeout = 20*1000
    print("Listening end")
try:
    # https://cloud.google.com/speech-to-text/docs/languages
    print("Speech by Google: "+ recognizer.recognize_google(audio, language ='en-US'))
    # See the other APIs

except sr.UnknownValueError:
    print("Error: Audio issue")
except sr.RequestError as e:
    print("Error: Speech Recognition service; {0}".format(e))
except:
    print("Error: Unknown")
    pass

#%% Speech Recognition using Audio Files
# reference: https://realpython.com/python-speech-recognition/
# reference: https://github.com/realpython/python-speech-recognition/blob/master/audio_files/harvard.wav

# Create recognizer object
recognizer=sr.Recognizer()

#http://www.voiptroubleshooter.com/open_speech/american.html
audio_file_source = sr.AudioFile('./data/OSR_us_000_0010_8k.wav')

#http://www.voiptroubleshooter.com/open_speech/india.html
# Change language (in recognize_google) when uncomment
#audio_file_source = sr.AudioFile('./data/OSR_in_000_0062_16k.wav')

#It records the data from the entire file into an AudioData instance.
with audio_file_source as source:
    # Use duration (in sec) for specific duration
    # Use offset to ignore from begining of record
    #audio = recognizer.record(source, offset=5, duration=10)

    #Enable only when you believe you have noisy audio. It reads the first second
    #of the file stream and calibrates the recognizer to the noise level of the audio.
    #Hence, that portion of the stream is consumed before you call record() to
    #capture the data.
    #recognizer.adjust_for_ambient_noise(source, duration=1)

    audio = recognizer.record(source)

type(audio)

try:
    # https://cloud.google.com/speech-to-text/docs/languages
    print("Speech by Google: "+recognizer.recognize_google(audio, language ='en-US')) #hi-IN en-US
    # See the other APIs

except sr.UnknownValueError:
    print("Error: Audio issue")
except sr.RequestError as e:
    print("Error: Speech Recognition service; {0}".format(e))
except:
    print("Error: Unknown")
    pass

# Few for noisy audio
#recognize_google(audio, show_all=True)

#CW: Take any other language and practice
#CW: Take your recording and practice
#CW: Take noisy audio and practice with/without recognizer.adjust_for_ambient_noise(source)

#%% Text to Speech
# Try your self with library pip install gTTS

#%% How to find similarity between two strings
#1. Cosine similarity (use for NLP)
#2. Jaccard similarity
#3. Levenshtein distance: The minimum number of single-character edits (insertions,
#deletions or substitutions) required to change one word into the other.
#https://en.wikipedia.org/wiki/Levenshtein_distance
#4. Hamming distance: Only for strings with equal length, the minimum number of
#substitutions required to change one string into the other
#https://en.wikipedia.org/wiki/Hamming_distance

from sklearn.metrics.pairwise import cosine_similarity

# Let us load reviews data which is Tf-IDF transformed
train = pd.read_csv("./data/Reviews_5000_cleaned_tfidf.csv")
train.columns = map(str.upper, train.columns)

#compute similarity for first sentence with rest of the sentences

# First make data in approprite format
first_row = train.iloc[0,2:].values
first_row.shape
first_row = np.reshape(first_row, (1, -1)) #it contains a single sample.

other_rows = train.iloc[1:,2:].values
other_rows.shape # ok

# Find similarity
cos_similarity = cosine_similarity(first_row,other_rows)
cos_similarity.min(), cos_similarity.max()

# CW: Try Jaccard, Levenshtein, Hamming at leisure time

#%% Language Translation
#Many libraries:
#pip install googletrans
#pip install translate
#pip install goslate
#pip install py-translate

import goslate

# Get translator object
gs_translator = goslate.Goslate()

# supported languages and their code
gs_translator.get_languages()

# test text
text = "I am learning NLP"
gs_translator.translate(text,'en')
gs_translator.translate(text,'hi')
gs_translator.translate(text,'fr')
gs_translator.translate(text,'de')

# detect language
id = gs_translator.detect('मैं एनएलपी सीख रहा हूं')
id
gs_translator.get_languages()[id]


