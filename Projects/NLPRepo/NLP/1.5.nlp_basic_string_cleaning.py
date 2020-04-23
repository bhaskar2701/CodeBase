#%% Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import time

import re
import string
from nltk.corpus import stopwords
from textblob import TextBlob
#from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

# Working directory
os.chdir("D:/trainings/NLP")

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value)

#%%  Read data in Panda data frame for explorations
train = pd.read_csv("./data/Reviews.csv")
train.columns = map(str.upper, train.columns)

# First view
train.shape
train.dtypes
train.info()

# Since dataype of text is 'object' hence converting to str
train['TEXT'] = train['TEXT'].astype(str)

# See the data
train.head()

# Let us see how many product data is available
prod_size = train.groupby('PRODUCTID').size()

# Sort
prod_size.sort_values(ascending  = False, inplace  = True)
prod_size[:5] # Note: B007JFMH8M has 913 rows. We will need this info in future

#For topmost product,  Keep only 'SCORE' and 'TEXT' only as these two are relevent for current analysis
train = train.loc[train['PRODUCTID'].isin(list(prod_size[:5].index)),['SCORE', 'TEXT']]
del(prod_size)

# If data count is huge and hence taking few data for each score
if train.shape[0] > 5000:
    df = pd.DataFrame()
    for score in train['SCORE'].unique(): # score = 1
        df = pd.concat([df, train[train['SCORE'] == score][:1000]], axis = 0) # rbind

    #get to train
    train = df; del(df, score)

train.reset_index(drop=True, inplace =True)

## Let us see the distribution of score
train['SCORE'].hist()
plt.show()

# Let us save records for viewing in XL
train.to_csv("./output/temp.csv",index = False)

# Let us see few text in above temp file or like here
train['TEXT'].head(100)
#What did you observe?

#%% Cleaning
# Make lower case
train['TEXT'] = train['TEXT'].str.lower()

# By default - do for each row
# Trim white spaces
train['TEXT'] = train['TEXT'].str.strip()

#For each line -> split in word -> remove extra whietspaces from word
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(word.strip().lower() for word in str_row_text.split()))

## remove unwanted characters, numbers and symbols
#train['TEXT'] = train['TEXT'].str.replace("[^a-zA-Z#]", " ")

# See how many rows contains '<br'
train[train['TEXT'].str.contains(r'<br />')].shape

# prepare list for replancing with whitespace
str_custom_replace_by_whitespaces = []
str_miscellaneous = ['<br />', 'Ã®ts', 'sooooo','each']; str_custom_replace_by_whitespaces.extend(str_miscellaneous)
str_number = ['one','two','three','four','five','six','seven','eight','nine','ten']; str_custom_replace_by_whitespaces.extend(str_number)
str_day = ['today','yesterday','tommorow']; str_custom_replace_by_whitespaces.extend(str_day)
str_amazon = ['amazon','amazoncom','amazoncomland','amazonits']; str_custom_replace_by_whitespaces.extend(str_amazon)
str_href = ['hrefhttpwwwamazoncomgpproductb000g6q4gmkettle','hrefhttpwwwamazoncomgpproductb001eo5qw8mccanns','hrefhttpwwwamazoncomgpproductb001gvisjmtwizzlers','hrefhttpwwwamazoncomgpproductb003tvdhiosmart','hrefhttpwwwamazoncomgpproductb004i3y4iegreen','httpwwwpremiumchocolatierscomingredientinfophp']; str_custom_replace_by_whitespaces.extend(str_href)
str_boolean = ['true','false']; str_custom_replace_by_whitespaces.extend(str_boolean)
str_double = ['coffeelattecappuccinoetc','couscousbrought']; str_custom_replace_by_whitespaces.extend(str_double)
str_months = ['january','february','march','april','may','june','july','august','september','october','november','december']; str_custom_replace_by_whitespaces.extend(str_months)

for str_custom in str_custom_replace_by_whitespaces:
    train['TEXT'] = train['TEXT'].str.replace(r'\b' + str_custom + r'\b',' ')

#Remove URL's
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: str_row_text.replace(r'http\S+', ''))
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: str_row_text.replace(r'ftp\S+', ''))

# Remove punctuations
for letter_punc in string.punctuation:
    train['TEXT'] = train['TEXT'].str.replace(letter_punc,'')

#For each line -> split in word -> keep words with more than length 2
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(word for word in str_row_text.split() if len(word) > 2))

# Search Engine skips common words. Details are at link
# https://www.shoutmeloud.com/seo-stop-words
list_search_engine_stopword = ["zero","you’ve","yourselves","yourself","yours","you’re","your","you’ll","you’d","you","yet","yes","wouldn’t","would","won’t","wonder","without","within","with","wish","willing","will","why","whose","who’s","whomever","whom","who’ll","whole","whoever","who’d","who","whither","whilst","while","whichever","which","whether","wherever","whereupon","where’s","wherein","whereby","whereas","whereafter","where","whenever","whence","when","what’ve","what’s","what’ll","whatever","what","we’ve","weren’t","we’re","were","went","we’ll","well","welcome","we’d","way","wasn’t","was","wants","want","viz","via","very","versus","various","value","usually","using","uses","useful","used","use","upwards","upon","unto","until","unlikely","unlike","unless","unfortunately","undoing","underneath","under","two","twice","t’s","trying","try","truly","tries","tried","towards","toward","took","too","together","till","thus","thru","throughout","through","three","though","those","thoroughly","thorough","this","thirty","third","think","things","thing","they’ve","they’re","they’ll","they’d","they","these","there’ve","thereupon","there’s","theres","there’re","there’ll","therein","therefore","there’d","thereby","thereafter","there","thence","then","themselves","them","theirs","their","the","that’ve","that’s","thats","that’ll","that","thanx","thanks","thank","than","tends","tell","taking","taken","take","sure","sup","such","sub","still","specifying","specify","specified","sorry","soon","somewhere","somewhat","sometimes","sometime","something","someone","somehow","someday","somebody","some","six","since","shouldn’t","should","she’s","she’ll","she’d","she","shan’t","shall","several","seven","seriously","serious","sent","sensible","selves","self","seen","seems","seeming","seemed","seem","seeing","see","secondly","second","says","saying","say","saw","same","said","round","right","respectively","relatively","regards","regardless","regarding","recently","recent","reasonably","really","rather","quite","que","provides","provided","probably","presumably","possible","plus","please","placed","perhaps","per","past","particularly","particular","own","overall","over","outside","out","ourselves","ours","our","oughtn’t","ought","otherwise","others","other","opposite","onto","only","one’s","ones","one","once","old","okay","often","off","obviously","nowhere","now","novel","notwithstanding","nothing","not","normally","nor","no-one","noone","nonetheless","none","non","nobody","ninety","nine","next","new","nevertheless","neverless","neverf","never","neither","needs","needn’t","need","necessary","nearly","near","namely","name","myself","mustn’t","must","much","mrs","mostly","most","moreover","more","miss","minus","mine","mightn’t","might","merely","meanwhile","meantime","mean","mayn’t","maybe","may","many","makes","make","mainly","made","ltd","lower","low","looks","looking","look","little","likewise","likely","liked","like","let’s","let","lest","less","least","latterly","latter","later","lately","last","knows","known","know","kept","keeps","keep","just","i’ve","itself","it’s","its","it’ll","it’d","isn’t","inward","into","instead","insofar","inside","inner","indicates","indicated","indicate","indeed","inc.","inc","inasmuch","immediate","i’m","i’ll","ignored","i’d","hundred","however","howbeit","how","hopefully","hither","his","himself","him","he’s","herself","hers","hereupon","here’s","herein","hereby","hereafter","here","her","hence","help","hello","he’ll","he’d","having","haven’t","have","hasn’t","has","hardly","happens","half","hadn’t","had","greetings","gotten","got","gone","going","goes","gives","given","getting","gets","get","furthermore","further","from","four","found","forward","forth","formerly","former","forever","for","follows","following","followed","five","first","fifth","fewer","few","farther","far","fairly","except","example","exactly","everywhere","everything","everyone","everybody","every","evermore","ever","even","etc","especially","entirely","enough","ending","end","elsewhere","else","either","eighty","eight","edu","each","during","downwards","down","don’t","done","doing","doesn’t","does","directly","different","didn’t","did","despite","described","definitely","daren’t","dare","currently","c’s","course","couldn’t","could","corresponding","contains","containing","contain","considering","consider","consequently","concerning","comes","come","com","co.","c’mon","clearly","changes","certainly","certain","causes","cause","caption","can’t","cant","cannot","can","came","but","brief","both","beyond","between","better","best","besides","beside","below","believe","being","behind","begin","beforehand","before","been","becoming","becomes","become","because","became","backwards","backward","back","awfully","away","available","associated","asking","ask","aside","a’s","around","aren’t","are","appropriate","appreciate","appear","apart","anywhere","anyways","anyway","anything","anyone","anyhow","anybody","any","another","and","amongst","among","amidst","amid","always","although","also","already","alongside","along","alone","almost","allows","allow","all","ain’t","ahead","ago","against","again","afterwards","after","adj","actually","across","accordingly","according","abroad","above","about","able"]
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(word for word in str_row_text.split() if not word in list_search_engine_stopword))

#Manual spelling correction
word_misspell =['looseleaf', 'middleeastern', 'aisl','akc']
word_correct =['loose leaf', 'middle eastern','','']
for count in range(len(word_misspell)):
    train['TEXT'] = train['TEXT'].str.replace(r'\b' + word_misspell[count] + r'\b',word_correct[count])

# Remove stopwords
# First see which all languages are available
stopwords.fileids()

# Let us get english specifc
set_stopwords = set(stopwords.words('english'))
set_stopwords

#For each line -> split in word -> include only if non stop word
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(word for word in str_row_text.split() if not word in set_stopwords))

#For each line -> split in word -> include only if non numeric digit
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(word for word in str_row_text.split() if not any(char.isdigit() for char in word)))
train['TEXT'].head(2)
# Spelling Correction: You have seen using TextBlob.
# Note: It will take many minutes
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: str(TextBlob(str_row_text).correct()))

## Do in batches as all in go is taking more time
#total_rows = train.shape[0]; batch_size = min([100, total_rows])
#for from_start in range(0, total_rows, batch_size):
#    start = time.time(); to_end = from_start+batch_size
#    if to_end > total_rows:
#        to_end = total_rows
#
#    print('%s%s%s%s' %('TextBlob Spelling Correction -> ', str(from_start), ' : ',str(to_end)))
#    train.loc[from_start:to_end, 'TEXT'] = train.loc[from_start:to_end, 'TEXT'].apply(lambda str_row_text: str(TextBlob(str_row_text).correct()))
#    print('%s%s' %('Time taken (min) ', str(round((time.time() - start)/60.0, 2))))
## end of for from_start in range(0, total_rows, batch_size):

# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
#For each line -> split in word -> lemma word
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(wordnet_lemmatizer.lemmatize(word) for word in str_row_text.split()))
#wordnet_lemmatizer.lemmatize('studi')

# Stemming
porter_stemmer = PorterStemmer()
#For each line -> split in word -> stem word
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(porter_stemmer.stem(word) for word in str_row_text.split()))
#porter_stemmer.stem('give')

#Try to remove word like 'mmmm' -> a word of same letter
# https://stackoverflow.com/questions/10072744/remove-repeating-characters-from-words
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(re.sub(r'(.)\1+', r'\1\1', word) for word in str_row_text.split()))

# After stemming, lemmentization few word amy become of 3 letter. Let us remove them
#For each line -> split in word -> keep words with more than length 3
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(word.strip().lower() for word in str_row_text.split() if len(word) > 3))

#Just precaution with long word - say greater than length 15
#For each line -> split in word -> keep words with less than length 15
train['TEXT'] = train['TEXT'].apply(lambda str_row_text: " ".join(word for word in str_row_text.split() if len(word) < 15))

#Final: Manual spelling correction
word_misspell =['gave','give','greenish']
word_correct =['','','green']
for count in range(len(word_misspell)):
    train['TEXT'] = train['TEXT'].str.replace(r'\b' + word_misspell[count] + r'\b',word_correct[count])

#for text in train['TEXT']:
#    if 'green' in text.split():
#        print(text)
#        break

#After cleaning, it may happen that anyline is left with 2 words. That may not be useful hence remove that line
#For each line -> split in word -> keep words with more than length 2
train['TEXT'].head(2)
train['TEXT_LEN'] = train['TEXT'].apply(lambda str_row_text: len(str_row_text.split()))
train = train[train['TEXT_LEN'] > 2]
train.drop('TEXT_LEN', axis=1, inplace=True)

# Let us save clean data for future use
train.to_csv("./data/Reviews_5000_cleaned.csv",index = False)

del(train, letter_punc, str_custom_replace_by_whitespaces, str_custom, list_search_engine_stopword, word_misspell, word_correct, set_stopwords, str_miscellaneous, str_number, str_day, str_amazon, str_href, str_boolean, str_double, str_months)

# Advice: One size does not suit all
# pip install pyspellchecker #from spellchecker import SpellChecker
# pip install autocorrect # from autocorrect import spell
# online/offline
