"""Given a set of cleaned data, this program creates a proprietary corpus"""
import os
# import webscraper
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import pandas as pd

p_cleaned_data = pd.read_csv('partially_cleaned_data.csv')
# Reverses the indexing in the dataframe to match the CaseID of each case
# p_cleaned_data.index = [x for x in range(webscraper.NUM_CASES, -1, -1)]
NUM_CASES = 1660
p_cleaned_data.index = [x for x in range(NUM_CASES, -1, -1)]

print(p_cleaned_data)

# Negation handling

# Handling other languages - This step must be done after lemmatisation
# as this library only stores the base of words
nltk.download('words') # Need to explicitly download the words corpus
english_words = set(words.words())

# Writing the corpus to a file

if __name__ == "__main__":
    # Loops through each piece of evidence for each case and operate on it
    updated_evidence = list(p_cleaned_data['evidence'])
    for item in updated_evidence:
        print(item)
        print("The type of the item is:", type(item))
        for i in range(0, len(item)):
            print(len(item))
            # TOKENISATION
            item[i]['title'] = word_tokenize(item[i]['title'])
            item[i]['description'] = word_tokenize(item[i]['description'])

            print(item[f"{i}"]['title'])

            # STOPWORD HANDLING - removes any stop words from the (now) tokenised evidence
            stop_words = set(stopwords.words('english'))

            # Converts the title and description into lists of strings
            # title_as_list = item[f"{i}"]['title'].split()
            # description_as_list = item[f"{i}"]['description'].split()

            # for j in range(0, len(title_as_list)):
            #     new_title = list() # The title with the stopwords removed
            #     if title_as_list[j] not in stop_words:
            #         new_title.append(title_as_list[j])

            # for k in range(0, len(description_as_list)):
            #     new_description = list() # The description with the stopwords removed
            #     if description_as_list[k] not in stop_words:
            #         new_description.append(description_as_list[k])

            # # LEMMATISATION - creates the lemmatisation object to turn the tokenised
            # # evidence into lemmas
            # lemma_maker = WordNetLemmatizer()


set(['jpeg',
    'gatekeeping',
    'realise',
    'realised',
    'delisting',
    'scammy',
    'filenames',
    'sergey',
    '6mb',
    '4k',
    'defi',
    'kleros',
    'png',
    'gif',
    'gifs',
    'suﬀicient',
    'rebalance',
    'reuploaded',
    'nasdaq',
    'web3',
    'filename',
    'app',
    'rehomed',
    'optimisation',
    'favicon',
    'webcam',
    'vpn',
    'doxing',
    'airbnb',
    'aggregator',
    'oversampled',
    'cofounder',
    'relisted',
    'dapp',
    'oﬄine',
    '4chan',
    'influencer',
    'aggregators',
    'untracked',
    'repurposed',
    'asynchrony',
    'presale',
    'whitegrey',
    'siri',
    'alexa',
    'petrov',
    'incentivized',
    'analytics',
    'gaslighting',
    'alexandr',
    'deepfakes',
    'waifu',
    'rebrand',
    'rebranding',
    'zhou',
    'requesters',
    'alessandro',
    'laggy',
    'jonny',
    'scammer',
    'scammers',
    'selfie',
    'emojis',
    'funder',
    'favour',
    'davide',
    'fernández',
    'decentralised',
    'misogynic',
    'requestor',
    'tiktok',
    'sunak',
    'transcoding',
    'mirred',
    'postediting',
    'binance',
    'delisted',
    'rebranded',
    'frontend',
    'tagline',
    'cristian',
    'reregister',
    'colours',
    'excludable',
    'zhang',
    'griefing',
    'mainnet',
    'maría',
    'weeklong',
    'pnk',
    'penalises',
    'fundraise',
    'web2',
    'imgur',
    'covid',
    'regularised',
    'rebalancing',
    'please',
    'hubristically',
    'backend',
    'unexhaustive',
    'jpg',
    'ethereum',
    'undoubtful',
    'cofounders',
    'retweets',
    'recognise',
    'widescreen',
    'kanye',
    'harbour',
    'upvoting',
    'unmirrored',
    'hijab',
    'tradability',
    'churros',
    'nitpicky',
    'gridlike',
    'selfies',
    'appliable',
    'resized',
    'upvote',
    'unstake',
    'webpages',
    'ruleset',
    'whitepaper',
    'memes',
    'favours',
    'disincentivizes',
    'freebooting',
    'coinbase',
    'depegging',
    'upvotes',
    'conspirations',
    'rebase',
    'urls',
    'bot',
    'pinterest',
    'dapps',
    'spammy',
    'devs',
    'reuploading',
    'unapologetically',
    'admins',
    'representability',
    'reupload',
    'inbox',
    'wellbeing',
    'behaviour',
    'recognised',
    'centralised',
    'resubmittal',
    'escrowed',
    'nunchucks',
    'rebalances',
    'searchable',
    'misfunction',
    'copypasting',
    'resize',
    'validator',
    'overlayed',
    'tokenlist',
    'centre',
    'linkedin',
    'kickstarter',
    'retweeting',
    'unslashed',
    'wiktionary',
    'preclaim',
    'iconified',
    'coindesk',
    'mysterium',
    'veriﬁed',
    'gmail',
    'requestors',
    'timelines',
    'decentralisation',
    'apologise',
    'bitcoin',
    'reddit',
    'customised',
    'retweet',
    'upvoted',
    'unconstructive',
    'themself',
    'organisation',
    'civilised',
    'pixelated',
    'hexlant',
    'mainpage',
    'ponzi',
    'preseeded',
    'retweeted',
    'borderless',
    'reposted',
    'readded',
    'duplicative',
    'uploader',
    'shroom',
    'momento',
    'recognising',
    'validators',
    'colour',
    'nano',
    'synchronisation',
    'litepaper',
    'funders',
    'unbanked',
    'repegging',
    'infographic',
    'depeg',
    'pixelates',
    'repost',
    'finalise',
    'misfunctions',
    'utf8',
    'bulleted',
    'explainers',
    'userbase']
    )