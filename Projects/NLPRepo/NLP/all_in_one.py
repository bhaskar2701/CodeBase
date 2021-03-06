import os, sys
import numpy as np
os.chdir("D://trainings//nlp")

try:
    exec(open(os.path.abspath('1.1.nlp_basic_string_operations.py'), encoding="utf8").read())
    print('1.1.nlp_basic_string_operations.py complete')
    exec(open(os.path.abspath('1.2.nlp_basic_regex.py'), encoding="utf8").read())
    print('1.2.nlp_basic_regex.py complete')
    #exec(open(os.path.abspath('1.3.nlp_basic_nltk_test.py'), encoding="utf8").read())
    #print('1.3.nlp_basic_nltk_test.py complete')
    exec(open(os.path.abspath('1.4.nlp_basic_nltk_operations.py'), encoding="utf8").read())
    print('1.4.nlp_basic_nltk_operations.py complete')
    exec(open(os.path.abspath('1.5.nlp_basic_string_cleaning.py'), encoding="utf8").read())
    print('1.5.nlp_basic_string_cleaning.py complete')
    exec(open(os.path.abspath('1.6.nlp_basic_wordcloud.py'), encoding="utf8").read())
    print('1.6.nlp_basic_wordcloud.py complete')
    exec(open(os.path.abspath('2.1.nlp_intermediate_entity_resolution.py'), encoding="utf8").read())
    print('2.1.nlp_intermediate_entity_resolution.py complete')
    exec(open(os.path.abspath('2.2.nlp_intermediate_text_to_features.py'), encoding="utf8").read())
    print('2.2.nlp_intermediate_text_to_features.py')
    exec(open(os.path.abspath('2.3.nlp_intermediate_word_embedding.py'), encoding="utf8").read())
    print('2.3.nlp_intermediate_word_embedding.py complete')
    exec(open(os.path.abspath('2.4.nlp_intermediate_operations.py'), encoding="utf8").read())
    print('2.4.nlp_intermediate_operations.py complete')
    exec(open(os.path.abspath('3.1.nlp_advance_classifications_ml.py'), encoding="utf8").read())
    print('3.1.nlp_advance_classifications_ml.py complete')
    exec(open(os.path.abspath('3.2.nlp_advance_classifications_dl.py'), encoding="utf8").read())
    print('3.2.nlp_advance_classifications_dl.py complete')
    exec(open(os.path.abspath('3.3.nlp_advance_sentiments.py'), encoding="utf8").read())
    print('3.3.nlp_advance_sentiments.py complete')
    exec(open(os.path.abspath('3.4.nlp_advance_clustering.py'), encoding="utf8").read())
    print('3.4.nlp_advance_clustering.py complete')
    exec(open(os.path.abspath('3.5.nlp_advance_topic_modeling.py'), encoding="utf8").read())
    print('3.5.nlp_advance_topic_modeling.py complete')
    exec(open(os.path.abspath('3.6.nlp_advance_search_engine.py'), encoding="utf8").read())
    print('3.6.nlp_advance_search_engine.py complete')

    print("All in One complete!")
except:
    print("Unexpected error: ", sys.exc_info()[0])
