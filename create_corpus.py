"""Given a set of cleaned data, this program creates a proprietary corpus"""
import os
import nltk
import data_cleaning
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
