""" Validates, cleans, normalises and processes data in the JSON format outlined in format.txt """
# Can be thought of as the normalisation part of the development process
import json
import pandas as pd
import re
import string
import emot
import csv
from spellchecker import SpellChecker
import os
import time
# import webscraper
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

# ---------------------------------------- PROGRAM FUNCTIONS ---------------------------------------- #
def remove_emojis(main_str_list, emoji_dict, emoticon_dict):
    """

    Args:
        main_str_list:
        emoji_dict:
        emoticon_dict:

    Returns:
        list(): The modified list of strings is returned.
    """
    if emoji_dict['flag'] is True or emoticon_dict['flag'] is True:
        emoji_counter = 0
        emoticon_counter = 0

        for a in range(0, len(main_str_list)):
            # Handling the emojis
            if main_str_list[a] in emoji_dict['value']:
                print("emoji:", main_str_list[a])
                main_str_list[a] = emoji_dict['mean'][emoji_counter]
                emoji_counter += 1
            # Handling the emoticons
            if main_str_list[a] in emoticon_dict['value']:
                print("emoticon:", main_str_list[a])
                main_str_list[a] = emoticon_dict['mean'][emoticon_counter]
                emoticon_counter += 1

    return main_str_list

def proprietary_preprocessing(df):
    """
    All necessary pre-processing that is done without the NLTK library

    Args:
        Takes the dataframe storing the JSON file of all the case data

    Returns:
        Returns the dataframe that has not been partially processed.
    """
    # Efficiently gets the case ID of each case where no evidence was found as a set of values
    no_evidence = list(df.loc[df['evidence'] == dict()].index)
    # Efficiently gets the case ID of each case where no data was found
    no_data = set(df.loc[(df['appeal'] == dict()) & (df['description'] == dict()) &\
                        (df['evidence'] == dict()) & (df['court'] == dict()) &\
                        (df['votes'] == dict()) & (df['start_date'] == dict())].index)

    print("The number of cases missing evidence are:", len(no_evidence))
    print("The number of cases with no data are:", len(no_data))

    # MANUAL COLLECTION FILLS THE ABOVE GAPS

    # --------------- DATA CLEANING/ PRE-PROCESSING STEPS ---------------
    # Removing duplicate cases is automatically done when the data is put into the data.json file

    # The altered evidence is stored in a standard Python list that,
    # which the existing evidence is then updated to
    updated_evidence = list(df['evidence'])

    # Extracts both the set of words to add to the pyspellchecker dictionary and the set of
    # chars to remove from the evidence
    with open('chars_processing.json', 'r') as data:
        charsets = json.load(data)

    # These are both stored in list format to be compatible with the methods being used
    chars_to_remove = charsets['chars_to_remove']
    words_to_add = charsets['extra_words']

    # Stores each of the words that pyspellcheck attempts to collect
    corrected_words = list()

    # As the evidence is stored as a dictionary, iterating over them is necessary
    for elem in updated_evidence:
        # Creates the object for detecting emojis/ emoticons in text
        e_obj = emot.core.emot()
        for i in range(0, len(elem)):
            # Use a regular expression to replace each of the escape characters (plus all \)
            # Removes from the evidence title -> removes the metadata
            elem[f"{i}"]['title'] = re.sub('[\\\\\"\n\r\t\b\f\a]',
                                        ' ', elem[f"{i}"]['title'])
            # Removes from the evidence description
            elem[f"{i}"]['description'] = re.sub('[\\\\\"\n\r\t\b\f\a]',
                                                 ' ', elem[f"{i}"]['description'])
            # The escape characters are replaced with spaces (this may make links easier to remove)

            # ... Then removes any links using regular expressions
            # TODO DO FOR DESCRIPTION/ OTHER TEXTUAL FIELDS
            elem[f"{i}"]['title'] = re.sub(r"http\S+", "", elem[f"{i}"]['title'])
            elem[f"{i}"]['description'] = re.sub(r"http\S+", "", elem[f"{i}"]['description'])

            # Creates a set of punctuation characters to remove
            chars_to_keep = set("'") # Keeps the apostrophe
            proprietary_char_removal = set(chars_to_remove)
            # Adds the identified chars to remove with the punctuation chars to remove (done as the union of the two sets)
            charset = proprietary_char_removal.union(set(string.punctuation).difference(chars_to_keep))

            # ... Then removes any punctuation characters -> There isn't a more efficient way to do this ?
            elem[f"{i}"]['title'] = ''.join(char for char in elem[f"{i}"]['title'] if char not in charset)
            elem[f"{i}"]['description'] = ''.join(char for char in elem[f"{i}"]['description'] if char not in charset)

            # Each step for the further pre-processing is done in a single loop to
            # increase efficiency (even if it doesn't look the nicest)
            # Getting the dictionary of slang words to use
            slang_file = open("slang.json")
            slang_dictionary = json.load(slang_file)
            slang_file.close()

            # Creating the spell checking object to use
            spell_check = SpellChecker(language='en')
            # Adds the additional words to the pyspellchecker dictionary
            spell_check.word_frequency.load_words(words_to_add)

            # Converts the title and description to lowercase
            elem[f"{i}"]['title'] = elem[f"{i}"]['title'].lower()
            elem[f"{i}"]['description'] = elem[f"{i}"]['description'].lower()

            # ---------------------------------------- FURTHER PRE-PROCESSING FOR THE TITLE ---------------------------------------- #
            # Initialises the emoji/ emoticon objects 
            title_emojis = e_obj.emoji(elem[f"{i}"]['title'])
            title_emoticons = e_obj.emoticons(elem[f"{i}"]['title'])
            # Break the title and description up into a list of words (split by whitespace)
            title_as_list = elem[f"{i}"]['title'].split()

            # Remove any emojis/ emoticons
            title_as_list = remove_emojis(title_as_list, title_emojis, title_emoticons)

            for j in range(0, len(title_as_list)):
                # Checks and expands the current word if it is identified as slang/ known jargon
                if title_as_list[j] in slang_dictionary.keys():
                    title_as_list[j] = slang_dictionary[title_as_list[j]]

            misspelt_words_t = spell_check.unknown(title_as_list)
            if len(misspelt_words_t) > 0:
                # Joins the title list into a single string
                elem[f"{i}"]['title'] = " ".join(title_as_list)
                for word in misspelt_words_t:
                    # Replaces the misspelt word in the title
                    correction = spell_check.correction(word)
                    # Validates that a correction can be found
                    if correction is not None:
                        elem[f"{i}"]['title'].replace(word, correction)
                        # Adds the misspelling to the list of corrections
                        corrected_words.append(word)
            else:
                # Join the description into a single string regardless
                elem[f"{i}"]['title'] = " ".join(desc_as_list)

            # ---------------------------------------- FURTHER PRE-PROCESSING FOR THE DESCRIPTION ---------------------------------------- #
            # Initialises the emoji/ emoticon objects
            desc_emojis = e_obj.emoji(elem[f"{i}"]['description'])
            desc_emoticons = e_obj.emoticons(elem[f"{i}"]['description'])
            # Breaks the description up into a list of words (split by whitespace)
            desc_as_list = elem[f"{i}"]['description'].split()

            # Remove any emojis/ emoticons
            desc_as_list = remove_emojis(desc_as_list, desc_emojis, desc_emoticons)

            for k in range(0, len(desc_as_list)):
                # Checks and expands the current word if it is identified as slang/ known jargon
                if desc_as_list[k] in slang_dictionary.keys():
                    desc_as_list[k] = slang_dictionary[desc_as_list[k]]
            
            misspelt_words_d = spell_check.unknown(desc_as_list)
            if len(misspelt_words_d) > 0:
                # Joins the description list into a single string
                elem[f"{i}"]['description'] = " ".join(desc_as_list)
                for word in misspelt_words_d:
                    # Replaces the misspelt word in the description
                    correction = spell_check.correction(word)
                    # Validates that a correction can be found
                    if correction is not None:
                        elem[f"{i}"]['description'].replace(word, correction)
                        # Adds the misspelling to the list of corrections
                        corrected_words.append(word)
            else:
                # Join the description into a single string regardless
                elem[f"{i}"]['description'] = " ".join(desc_as_list)

    # Removes duplicates from the list
    corrected_words = list(set(corrected_words))
    # Writes the set of corrected words to a csv file
    with open("special_words.txt", 'w') as txtfile:
        # Write each word like an index in a list ['x', 'y', ..., 'z']
        txtfile.writelines("[")
        for item in corrected_words:
            if item == corrected_words[-1]:
                txtfile.writelines(item)
            else:
                txtfile.writelines(item + ", ")
        
        txtfile.writelines("]")

    # Updates the dataframe to contain the updated evidence
    df['evidence'] = updated_evidence

    return df

def corpus_creation(df):
    """
    Performs any NLTK operations and data cleaning and creates a corpus
    from the provided case data.

    Args:

    Returns:

    """
    # Like in the proprietary_preprocessing function, this extracts the
    # evidence tab from the dataframe and turns it into a list of strings
    evidence = list(df['evidence'])

    # Stores each of the words used in the title
    title_data = set()
    # Stores each of the words used in the description
    description_data = set()
    # Sets are used to prevent any duplicates from being added

    for elem in evidence:
        for i in range(0, len(elem)):
            # Stopword handling
            stop_words = set(stopwords.words('english'))
            title_as_list = elem[f"{i}"]['title'].split()
            desc_as_list = elem[f"{i}"]['description'].split()

            # Removes the stopwords from the title
            new_title = list() # The updated title 
            for j in range(0, len(title_as_list)):
                if title_as_list[j] not in stop_words:
                    new_title.append(title_as_list[j])

            # Removes the stopwords from the description
            new_desc = list()
            for k in range(0, len(desc_as_list)):
                if desc_as_list[k] not in stop_words:
                    new_desc.append(desc_as_list[k])

            # Updates, recombines and tokenises both lists for the evidence
            elem[f"{i}"]['title'] = word_tokenize(" ".join(new_title))
            elem[f"{i}"]['description'] = word_tokenize(" ".join(new_desc))

            # Creates the lemmatisation object and 'lemmatises' each (now) processed token
            lemma_maker = WordNetLemmatizer()

            # Lemmatising the title
            title_tokens = elem[f"{i}"]['title']
            for x in range(0, len(title_tokens)):
                title_tokens[x] = lemma_maker.lemmatize(title_tokens[x])

            # Lemmatising the description
            desc_tokens = elem[f"{i}"]['description']
            for y in range(0, len(desc_tokens)):
                desc_tokens[y] = lemma_maker.lemmatize(desc_tokens[y])

            # Updates the title and description with the lemmatised tokens
            elem[f"{i}"]['title'] = title_tokens
            elem[f"{i}"]['description'] = desc_tokens

            for item in elem[f"{i}"]['title']:
                title_data.add(item)

            for item in elem[f"{i}"]['description']:
                description_data.add(item)

    # Converts the set into a list (so it is basically a tokenised list)
    title_data = list(title_data)
    description_data = list(description_data)

    # With the data organised, put the data into two separate files
    dir = "kleros_corpora"
    # Creates a directory for the corpora if none exists
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Write the tokenised data to two separate files in this new directory
    with open(f"{dir}/title_corpus.txt", "w", encoding="utf-8") as file:
        file.write(" ".join(title_data))
    
    with open(f"{dir}/description_corpus.txt", "w", encoding="utf-8") as file:
        file.write(" ".join(description_data))

    # Creates an object to read the newly created plain text corpora and outputs a sample
    corpus = PlaintextCorpusReader(dir, ".*\.txt")
    print("Sample words from both corpora:", corpus.words())
    

if __name__ == "__main__":
    # Read the JSON case data into a Pandas dataframe (using the CaseID as the index)
    df = pd.read_json("data.json", orient='index')

    # Performs all non-NLTK operations on the dataframe
    start = time.time()
    df = proprietary_preprocessing(df)
    print("Time taken to perform proprietary pre-processing:", str(time.time() - start) + "s")

    corpus_creation(df)

    # Once the data has been cleaned and the corpus created -> compare to VADER lexicon and calculate the polarity score for words not in the sentiment lexicon

    # Then (IN A SEPARATE FILE) -> follow on from stance classification paper and use the dataset they also use -> Train a sentiment analysis classifier. (No feature extraction necessary??)
