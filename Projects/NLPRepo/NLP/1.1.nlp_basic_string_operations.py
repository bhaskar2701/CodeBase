# Note: NO library loaded

# get some test data
str_text = "I am learning NLP. We will start with basic where we will learn various things \
- String operation, Stop words, Tokenization, Stemming, Lemmantisation, Spelling correction,\
Parts of speech tagging. "

# Extract using index and range
str_text[0]
str_text[0:4]
str_text[:10]
str_text[10:]
str_text[-1]
str_text[-2]
str_text[-3]
str_text[-3:]

# To make lower
str_text.lower()

# To make upper
str_text.upper()

#titlecased version
str_text.title() # Notice cap letter for each word

#Remove leading or trailing whitespace, if any
str_text.strip()

#split into a list
str_text.split() # usages whitespace by default
str_text.split('.') # using custom '.'

#join the words into a string using first as glue
' '.join(str_text.split()) # join by whitespace
', '.join(str_text.split()) # join by comma and whitespace

#index of first instance (-1 if not found)
str_text.lower().find('we')

#index of last instance (-1 if not found)
str_text.lower().rfind('we')

#replace instances of first with second
str_text.lower().replace('will', 'shall')
