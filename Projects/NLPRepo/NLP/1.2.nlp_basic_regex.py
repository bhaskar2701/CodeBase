#Load library
import re

# get some test data
str_text = "I am learning NLP. We will start with basic where we will learn various things \
- String operation, Stop words, Tokenization, Stemming, Lemmantisation, Spelling correction,\
Parts of speech tagging. "

#match of the string only at the beginning of the string else returns a none
re.match('we', str_text.lower())

#match anywhere in the string although will return the first occurence
re.search('we', str_text.lower())

# Find all occurences
re.findall('we', str_text.lower())

#Replace "i" with "I"
re.sub('will', 'shall', str_text)

#find all occurance of text in the format "abc, xyz"
re.findall(r'[a-zA-Z0-9]*, [a-zA-Z0-9]*', str_text)

#See the help of any of above and you will see 'flag' param having follwoing info
#The basic flags are I, L, M, S, U, X:
#re.I: This flag is used for ignoring casing
#re.M: This flag is useful if you want to find patterns throughout multiple lines
#re.L: This flag is used to find a local dependent
#re.S: This flag is used to find dot matches
#re.U: This flag is used to work for unicode data
#re.X: This flag is used for writing regex in a more readable format

#CW: Practice at leisure
#Find the single occurrence of character a and b: Regex: [ab]
#Find characters except a and b: Regex: [^ab]
#Find the character range of a to z: Regex: [a-z]
#Find range except to z: Regex: [^a-z]
#Find all the characters a to z as well as A to Z: Regex: [a-zA-Z]
#Any single character: Regex: .
#Any whitespace character: Regex: \s
#Any non-whitespace character: Regex: \S
#Any digit: Regex: \d
#Any non-digit: Regex: \D
#Any non-words: Regex: \W
#Any words: Regex: \w

#CW: Visit at leisure https://docs.python.org/2/library/re.html

# Advice: practice