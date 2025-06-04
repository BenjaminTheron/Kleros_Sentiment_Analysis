""" Validates, cleans, normalises and processes data in the JSON format outlined in format.txt """
# Can be thought of as the normalisation part of the development process
import json
import pandas as pd
import re
import string
import emot
from spellchecker import SpellChecker

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

# Validating the data to check what data was missed during the collection process.
no_evidence = list()
missing_data = list()

# Open the data JSON file
file = open('data.json')

# Read the json file into a dictionary
data = json.load(file)

# Close the file as soon as it's no longer needed
df = pd.read_json("data.json", orient='index')

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

# As the evidence is stored as a dictionary, iterating over them is necessary
for elem in updated_evidence:
    # Creates the object for detecting emojis/ emoticons in text
    e_obj = emot.core.emot()
    for i in range(0, len(elem)):
        # Use a regular expression to replace each of the escape characters (plus all \)
        # Removes from the evidence title
        elem[f"{i}"]['title'] = re.sub('[\\\\\"\n\r\t\b\f\a]',
                                       ' ', elem[f"{i}"]['title'])
        # Removes from the evidence description
        elem[f"{i}"]['description'] = re.sub('[\\\\\"\n\r\t\b\f\a]',
                                             ' ', elem[f"{i}"]['description'])
        # The escape characters are replaced with spaces (this may make links easier to remove)

        # Converts to lowercase
        elem[f"{i}"]['title'] = elem[f"{i}"]['title'].lower()
        elem[f"{i}"]['description'] = elem[f"{i}"]['description'].lower()

        # ... Then removes any links using regular expressions
        # TODO DO FOR DESCRIPTION/ OTHER TEXTUAL FIELDS
        elem[f"{i}"]['title'] = re.sub(r"http\S+", "", elem[f"{i}"]['title'])
        elem[f"{i}"]['description'] = re.sub(r"http\S+", "", elem[f"{i}"]['description'])

        # Creates a set of punctuation characters to remove
        chars_to_keep = set("'") # Keeps the apostrophe
        charset = set(string.punctuation).difference(chars_to_keep)

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
                print("Slang being expanded:", title_as_list[j])
                title_as_list[j] = slang_dictionary[title_as_list[j]]
            
            # Corrects the current word if it is misspelt
            # correction = spell_check.correction(title_as_list[j])
            # if correction is not None and title_as_list[j] != correction:
            #     print("Current word:", title_as_list[j], ".Corrected word:", correction)
            #     title_as_list[j] = correction

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
                print("Slang being expanded:", desc_as_list[k])
                desc_as_list[k] = slang_dictionary[desc_as_list[k]]
            
            # Corrects the current word if it is misspelt
            # correction = spell_check.correction(desc_as_list[k])
            # if correction is not None and desc_as_list[k] != correction:
            #     print("Current word:", desc_as_list[k], ".Corrected word:", correction)
            #     desc_as_list[k] = correction

        # Recombines both lists of strings into a the final pre-processed evidence string
        elem[f"{i}"]['title'] = " ".join(title_as_list)
        elem[f"{i}"]['description'] = " ".join(desc_as_list)

# Updates the dataframe to contain the updated evidence
df['evidence'] = updated_evidence

print(df['evidence'][1366])

# NLTK IS USED FOR THE BELOW IN THE CREATE CORPUS FILE.
# Stopword handling

# Negation handling

# Handling other languages

# Tokenisation

# Lemmatisation