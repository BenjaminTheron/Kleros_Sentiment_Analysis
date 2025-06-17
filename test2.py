import string
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

sentence = "this is a sentence containing some words, i.e. ethereum address, proof of humanity registry, incorrect picture"

# remove the punctuation
sentence = ''.join(char for char in sentence if char not in string.punctuation)

# remove the stopwords
words_list = word_tokenize(sentence)

# Creates a sentiment intensity analyser
sia = SentimentIntensityAnalyzer()

vader_dict = sia.lexicon

print(vader_dict)

print(f"The length of the vader dictionary is {len(vader_dict)}")

print(f"The words list is {words_list}")

# remove stopwords and lemmatise sentence
updated_words_list = list()
for i in range(0, len(words_list)):
    if words_list[i] not in stopwords.words('english'):
        updated_words_list.append(words_list[i])

# lemmatise the remaining words

for word in updated_words_list:
    if word in vader_dict:
        print(f"{word} in the vader dictionary, with score: {vader_dict[word]}")
    else:
        print(f"{word} is NOT in the vader dictionary")