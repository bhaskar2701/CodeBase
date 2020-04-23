#%% Tokenizers
# On ppt 'Challenges of sentence tokenization'

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Sentence tokenizer
text = "this's a sent tokenize test. this is sent two. is this sent three? sent 4 is cool! Now it is your turn."
sent_tokenize_list = sent_tokenize(text) # Uses PunktSentenceTokenizer trained on english. There are pretrained tokenizers for many languages.
len(sent_tokenize_list) #5
sent_tokenize_list

#Tokenizing text into words
word_tokenize(text) # Uses TreebankWordTokenizer

#Class work: Use the MWE (multi-word expression) tokenizer for "multi word - phrases"
#from nltk.tokenize.mwe import MWETokenizer

# CW: One '1.5.nlp_basic_string_cleaning.py' is done then use file 'Reviews_5000_cleaned.csv' and print 5 words with highest frequency

#%% Part-of-speech tagging
import nltk
text = "This is python training by Shiv for the Analytics team happening at Bangalore . he is felicitating and helping us ."
nltk.pos_tag(text.split())

#NLTK provides documentation for each tag
nltk.help.upenn_tagset("RB")
nltk.help.upenn_tagset("NN*")
nltk.help.upenn_tagset("JJ")
nltk.help.upenn_tagset("IN")

#%% Stemming and Lemmatization
# Definition on ppt

# NLTK provides several famous stemmers interfaces, such as Porter stemmer, Snowball Stemmer
#Lancaster Stemmer, .The aggressiveness continues basically following along those same lines

#For Porter Stemmer, which is based on The Porter Stemming Algorithm, can be used like this

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

# Examples
porter_stemmer.stem("study")
porter_stemmer.stem("studies")
porter_stemmer.stem("studying")
porter_stemmer.stem("maximum")
porter_stemmer.stem("presumably")
porter_stemmer.stem("multiply")
porter_stemmer.stem("provision")
porter_stemmer.stem("owed")
porter_stemmer.stem("ear")
porter_stemmer.stem("saying")
porter_stemmer.stem("crying")
porter_stemmer.stem("string")
porter_stemmer.stem("meant")
porter_stemmer.stem("cement")

#For Snowball Stemmer, which is based on Snowball Stemming Algorithm, can be used in NLTK like this:
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
snowball_stemmer.stem("study")
snowball_stemmer.stem("studies")
snowball_stemmer.stem("studying")
snowball_stemmer.stem("maximum")
snowball_stemmer.stem("presumably")
snowball_stemmer.stem("multiply")
snowball_stemmer.stem("provision")
snowball_stemmer.stem("owed")
snowball_stemmer.stem("ear")
snowball_stemmer.stem("saying")
snowball_stemmer.stem("crying")
snowball_stemmer.stem("string")
snowball_stemmer.stem("meant")
snowball_stemmer.stem("cement")

#For Lancaster Stemmer, which is based on The Lancaster Stemming Algorithm
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
lancaster_stemmer.stem("study")
lancaster_stemmer.stem("studies")
lancaster_stemmer.stem("studying")
lancaster_stemmer.stem("maximum")
lancaster_stemmer.stem("presumably")
lancaster_stemmer.stem("presumably")
lancaster_stemmer.stem("multiply")
lancaster_stemmer.stem("provision")
lancaster_stemmer.stem("owed")
lancaster_stemmer.stem("ear")
lancaster_stemmer.stem("saying")
lancaster_stemmer.stem("crying")
lancaster_stemmer.stem("string")
lancaster_stemmer.stem("meant")
lancaster_stemmer.stem("cement")

#How to use Lemmatizer in NLTK. The NLTK Lemmatization method is based on WordNet's built-in morphy
# function. In NLTK, you can use it as the following:
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer.lemmatize("study")
wordnet_lemmatizer.lemmatize("studies")
wordnet_lemmatizer.lemmatize("studying")
wordnet_lemmatizer.lemmatize("dogs")
wordnet_lemmatizer.lemmatize("churches")
wordnet_lemmatizer.lemmatize("aardwolves")
wordnet_lemmatizer.lemmatize("abaci")
wordnet_lemmatizer.lemmatize("hardrock")
wordnet_lemmatizer.lemmatize("are")
wordnet_lemmatizer.lemmatize("is")
#You would note that the "are” and "is” lemmatize results are not "be”, that"s because the lemmatize
#method default pos argument is "n”. v(verb), a(adjective), r(adverb), n(noun).

#So you need specified the pos for the word like these:
wordnet_lemmatizer.lemmatize("is", pos="v")
wordnet_lemmatizer.lemmatize("are", pos="v")

#%% Word Sense Disambiguation
# Theory and challenges on ppt

from nltk.wsd import lesk
sent = ['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.']
print(lesk(sent, 'bank', pos = 'n')) # See the meaning in SYsnet at bottom of this section
print(lesk(sent, 'bank'))

sent = ['I', 'went', 'to', 'the', 'bank', 'to', 'catch', 'ship', '.']
print(lesk(sent, 'bank', pos = 'n'))
print(lesk(sent, 'bank'))

from nltk.corpus import wordnet as wn
for ss in wn.synsets('bank'):
    print(ss, ss.definition())

#%% Calculate BLEU Scores. Bilingual Evaluation Understudy, is a score for comparing a
#candidate translation of text to one or more reference translations.
#The Python Natural Language Toolkit library, or NLTK, provides an implementation of
#the BLEU score that you can use to evaluate your generated text against a reference.

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

#Sentence BLEU Score
#NLTK provides the sentence_bleu() function for evaluating a candidate sentence
#against one or more reference sentences.

#The reference sentences must be provided as a list of sentences where each reference
#is a list of tokens. The candidate sentence is provided as a list of tokens.

reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score) # Running this example prints a perfect score as the candidate matches one of the references exactly.

candidate = ['this', 'was', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score) # .70

#Corpus BLEU Score: NLTK also provides a function called corpus_bleu() for calculating the BLEU
#score for multiple sentences such as a paragraph or a document.

#The references must be specified as a list of documents where each document is a list of references
# and each alternative reference is a list of tokens, e.g. a list of lists of lists of tokens. The
# candidate documents must be specified as a list where each document is a list of tokens, e.g. a
# list of lists of tokens.

# two references for one document
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
print(score) # Running the example prints a perfect score as before.

candidates = [['this', 'was', 'a', 'test']]
score = corpus_bleu(references, candidates)
print(score) # .70

#Cumulative and Individual BLEU Scores
#The BLEU score calculations in NLTK allow you to specify the weighting of different n-grams in the
# calculation of the BLEU score.
#
#Individual N-Gram Scores
#An individual N-gram score is the evaluation of just matching grams of a specific order, such as
#single words (1-gram) or word pairs (2-gram or bigram).
#
#The weights are specified as a tuple where each index refers to the gram order. To calculate the
#BLEU score only for 1-gram matches, you can specify a weight of 1 for 1-gram and 0 for 2, 3 and 4
#(1, 0, 0, 0). For example:

# 1-gram individual BLEU
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score) # 0.75

#Cumulative N-Gram Scores
#Cumulative scores refer to the calculation of individual n-gram scores at all orders from 1 to n
#and weighting them by calculating the weighted geometric mean.
#
#By default, the sentence_bleu() and corpus_bleu() scores calculate the cumulative 4-gram BLEU
#score, also called BLEU-4.
#
#The weights for the BLEU-4 are 1/4 (25%) or 0.25 for each of the 1-gram, 2-gram, 3-gram and 4-gram scores. For example:

# 4-gram cumulative BLEU
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)

#The cumulative and individual 1-gram BLEU use the same weights, e.g. (1, 0, 0, 0). The 2-gram
#weights assign a 50% to each of 1-gram and 2-gram and the 3-gram weights are 33% for each of the 1,
# 2 and 3-gram scores.

# cumulative BLEU scores
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

#It is common to report the cumulative BLEU-1 to BLEU-4 scores when describing the skill of a text
#generation system.

# CW: play and practice. Observe the differences

# prefect match
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

#Next, let’s change one word, ‘quick‘ to ‘fast‘.
# one word different
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score) # 0.75, This result is a slight drop in score.

#Try changing two words, both ‘quick‘ to ‘fast‘ and ‘lazy‘ to ‘sleepy‘.
# two words different
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
print(score) # 0.48, linear drop in skill.

#What about if all words are different in the candidate?
# all words different
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
score = sentence_bleu(reference, candidate)
print(score) # 0

#Now, let’s try a candidate that has fewer words than the reference (e.g. drop the last two words),
# but the words are all correct.
# shorter candidate
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the']
score = sentence_bleu(reference, candidate)
print(score) # 0.75, The score is much like the score when two words were wrong above.

#How about if we make the candidate two words longer than the reference?
# longer candidate
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'from', 'space']
score = sentence_bleu(reference, candidate)
print(score) # 0.78, Again, the score is something like “two words wrong“.

#Finally, let’s compare a candidate that is way too short: only two words in length.
# very short
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick']
score = sentence_bleu(reference, candidate)
print(score) # 0.03
#Running this example first prints a warning message indicating that the 3-gram and above part of
#the evaluation (up to 4-gram) cannot be performed. This is fair given we only have 2-grams to work
# with in the candidate.

#%% CW: Explore in detail - StanfordNLP
#https://nlp.stanford.edu/

#Stanford : Named-entity recognition (NER)
from nltk.tag.stanford import StanfordNERTagger
import nltk

# Locate nltk_data folder and provide path
#st = StanfordNERTagger("D:\\nltk_data\\stanford-ner-2014-06-16\\classifiers\\english.all.3class.distsim.crf.ser.gz", "D:\\nltk_data\\stanford-ner-2014-06-16\\stanford-ner.jar")
st = StanfordNERTagger("D:\\nltk_data\\stanford-ner-2015-04-20\\classifiers\\english.all.3class.distsim.crf.ser.gz", "D:\\nltk_data\\stanford-ner-2015-04-20\\stanford-ner.jar")
text = "This is python training by Shiv for the Analytics team happening at Bangalore. he is felicitating and helping us ."
st.tag(nltk.word_tokenize(text))
# Notice the mistake in identification

#%%TextBlob
from textblob import TextBlob

text = "Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models."
nlpblob = TextBlob(text)

#introduce various features of TextBlob

#1) Word Tokenization
nlpblob.words

#2) Sentence Tokenization
nlpblob.sentences

#3）Part-of-Speech Tagging
nlpblob.tags

#4) Noun Phrase Extraction
nlpblob.noun_phrases

#5) Sentiment Analysis
nlpblob.sentiment.polarity # polarity is a value between -1.0 and +1.0
nlpblob.sentiment.subjectivity # subjectivity between 0.0 and 1.0.
nlpblob.sentiment

#sentiment polarity
for sentence in nlpblob.sentences:
    print(sentence.sentiment.polarity)

#6) Word Singularize
nlpblob.words[138]
nlpblob.words[138].singularize()

#7) Word Pluralize
nlpblob.words[21]
nlpblob.words[21].pluralize()

#8) Words Lemmatization
#Words can be lemmatized by the lemmatize method, but notice that the TextBlog lemmatize method is
# inherited from NLTK Word Lemmatizer, and the default POS Tag is "n", if you want lemmatize other
#pos tag words, you need specify it:
nlpblob.words[138].pluralize().lemmatize()
nlpblob.words[21].pluralize().lemmatize()

#9）Spelling Correction
#TextBlob Spelling correction is based on Peter Norvig"s "How to Write a Spelling Corrector", which is
# implemented in the pattern library:
b = TextBlob("I havv good speling!")
b.correct()

#Word objects also have a spellcheck() method that returns a list of (word, confidence) tuples with spelling suggestions:

#9) Parsing: TextBlob parse method is based on pattern parser:
nlpblob.parse()

#10) Translation and Language Detection: By Google"s API:
#Detect
nlpblob.detect_language()

nlpblob.translate(to="hi")
nlpblob.translate(to="kn") # es fr
nlpblob.translate(to="fr") # es fr
nlpblob.translate(to="zh")

# Few more example. How to get keyword for any particular language
non_eng_blob = TextBlob("हिन्दी समाचार की आधिकारिक वेबसाइट. पढ़ें देश और दुनिया की ताजा ख़बरें")
non_eng_blob.detect_language()

non_eng_blob = TextBlob("ಮುಖ್ಯ ವಾರ್ತೆಗಳು ಜನಪ್ರಿಯ")
non_eng_blob.detect_language()
non_eng_blob.translate(to="en")
non_eng_blob.translate(to="hi")

#Class work: Try your native languages

#%% CW: Explore FastText (An NLP library by Facebook)
#http://feedproxy.google.com/~r/AnalyticsVidhya/~3/r-TzzESKAbQ/?utm_source=feedburner&utm_medium=email

#%%Pattern is a web mining module for the Python programming language.
#It has tools for data mining (Google, Twitter and Wikipedia API, a web crawler, a HTML DOM parser),
#natural language processing (part-of-speech taggers, n-gram search, sentiment analysis, WordNet), machine learning
# (vector space model, clustering, SVM), network analysis and canvas visualization.
#It is  text data mining tool which including crawler
# from pattern.en import *
