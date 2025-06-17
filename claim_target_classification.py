""" Contains all the functionality required to identify the target of a claim """
import os
import pandas as pd
import nltk
import string
import stanza
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def breadth_first_search(tree, target):
    """
    A proprietary implementation of the breadth first algorithm to search
    tree data structures, this variant searches for a given target in the
    specific tree structure returned by a Stanza syntactic parse of a sentence.
    Furthermore, this algorithm doesn't stop at the first target it finds (ultimately
    traverses the entire syntax tree).

    phrases can contain other phrases

    Args:

    Returns:
    """
    # Contains each noun phrase in the tree as a tokenised list
    noun_phrases = list()
    # 'Queue' used to track which nodes have been visited
    queue = list()
    # Adds the root node to the queue
    queue.append(tree)

    # Tracks the current index in the list (the head of the queue)
    # Uses a sliding window technique instead of removing the values (simplicity sake)
    index = 0
    while index != len(queue):
        # Visits the current node (checks if it's an NP)
        if queue[index].label == target:
            # Need to perform a DEPTH FIRST TRAVERSAL to extract the words for each phrase
            new_root = queue[index]
            # Stores the nodes which have been visited
            to_visit = []
            # Adds the root to the list of visited nodes
            to_visit.append(new_root)
            # Stores the resulting sentence
            phrase = list()

            # DON'T need to keep track of the nodes that have been visited (no overlap in these graphs)
            # Search until there are no more nodes to visit
            while len(to_visit) > 0:
                # Stack structure is used (look at the elements on the rightmost side of the list)
                current_node = to_visit.pop()
                # If the current node has no children, then it must be a word
                if len(current_node.children) == 0:
                    # The node is still stored as a node so it must be converted into a string
                    phrase.append(str(current_node))
                else:
                    # Add the children to the stack and keep going down the tree
                    for i in range(len(current_node.children) - 1, -1, -1):
                        to_visit.append(current_node.children[i])

            # Adds the tokenised phrase to the list of phrases
            noun_phrases.append(" ".join(phrase))

        # Adds the children of the current node to the queue to be visited
        for children in queue[index].children:
            queue.append(children)
        
        index += 1
        
    # Returns the list of NPs (which still need to be expanded)
    return noun_phrases


# Generate a file with all the case descriptions and case titles split by sentence
def create_sentence_data(df, file_path):
    """

    Args:

    Returns:
    """
    # Initialises the English neural network pipeline
    nlp_pipeline = stanza.Pipeline('en', processors='tokenize,mwt,pos,constituency')

    if not os.path.isfile(file_path):
        # Breaks all the case data (title + desc) down into sentences to be labelled
        # Extracts the evidence data
        evidence_data = list(df['evidence'])
        description_data = list(df['description'])
        evidence_dict = {}

        for i in range(0, len(evidence_data)):
            current_case = evidence_data[i]
            data_dict = dict()
            # Adds the case description to the dictionary to add more context to the true target
            description_analyser = nlp_pipeline(description_data[i])
            largest = ""
            # The largest NP in the description is taken as the True Target
            for sentence in description_analyser.sentences:
                for NP in breadth_first_search(sentence.constituency, "NP"):
                    if len(list(NP)) > len(list(largest)):
                        largest = NP

            data_dict["description"] = {
                "description": description_data[i],
                "target": largest
            }
            # Stores the data and corresponding labels for each sentence in a given case
            claim = dict()
            for j in range(0, len(current_case)):
                # Titles and descriptions are stored in their full form (not tokenised into sentences)
                # However they are stored as one (not treated differently) dataset
                case_text = current_case[f"{j}"]["title"] + ". " +\
                            current_case[f"{j}"]["description"]
                
                # Feeds the case_text to the NN pipeline to perform syntactic operations
                text_analyser = nlp_pipeline(case_text)
                # Stores the NPs in the given piece of text
                NPs = list()
                for sentence in text_analyser.sentences:
                    # Performs a syntactic parse (via Stanza) to get the NPs
                    # Then extracts these using a BFS on the syntax tree
                    NPs.append(breadth_first_search(sentence.constituency, "NP"))
                
                # Logs the relevant data for the current claim
                claim[f"{j}"] = {
                    "text": case_text,
                    "candidate_phrase(s)": NPs,
                    "labelled_target": "", # This is a manually prescribed target phrase
                    "overlap": "",
                    "best_phrase": "",
                    "example_type": ""
                }

            data_dict["claims"] = claim

            evidence_dict[f"{i}"] = data_dict
            print(f"claim{i} logged...")
            # Displays the parse tree for one of cases (currently CaseID 1660)
            
        # Write the evidence dictionary to a JSON file
        with open(file_path, "w") as file:
            json.dump(evidence_dict, file, indent=4)

    # If the file exists, it is assumed that the data in the file is correctly
    # formatted/ already exists. (Done to prevent the manual labels being overwritten)

# Calculates the Jaccard Index
def jaccard_index(set_a, set_b):
    """
    The Jaccard Index calculates the amount of overlap between two phrases, this is measured
    as the size of the intersection divided by the size of the union of the two sets.

    Args:

    Returns:

    """
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    # This is given to two decimal places
    return round((len(intersection) / len(union)), 2)


def overlap(phrase_a, phrase_b):
    """
    Using the functionality outlined in the jaccard_index function, this function
    computes the amount of overlap between two phrases and determines whether
    the given phrase is a positive or negative example for the logistic regression
    classifier.

    Args:

    Returns: Returns a tuple containing the overlap score and the type of example
             indicated by this score.
    """
    # Currently using the threshold for positive/ negative example labels outlined
    # in the IBM paper of > 0.6 for positive, otherwise negative
    THRESHOLD = 0.6

    # Converts the given text phrases into sets (first pre-processes the strings to ensure
    # good comparability -> lowercasing and punctuation removal)
    phrase_a = ''.join(char for char in phrase_a if char not in string.punctuation).lower()
    phrase_b = ''.join(char for char in phrase_b if char not in string.punctuation).lower()

    # Tokenises both phrases so we get a word by word comparison using the Jaccard index
    phrase_a = word_tokenize(phrase_a)
    phrase_b = word_tokenize(phrase_b)

    # Converts the phrases into sets to meet the requirements of the function
    # This also ensures that misspelt words will not tank the overlap between phrases
    # TODO CHANGE TO MAKE A COMPARISON BETWEEN THE SET OF TOKENS NOT INDIVIDUAL CHARACTERS
    score = jaccard_index(set(phrase_a), set(phrase_b))
    example_type = "POSITIVE" if score >= THRESHOLD else "NEGATIVE"

    return (score, example_type)


def label_data(data_file, output_file):
    """
    Extracts the claims data in the provided data file, and for each claim in the data
    it finds the overlap between the provided label and each candidate phrase, with the
    phrase containing the highest overlap score having itself, its overlap score and the
    corresponding example type labelled in a dictionary which is written to a new json
    file.

    Args:
    data file -> the json file to extract the claims data from
    output file -> the json file to write the data to.


    Returns:
        The phrase features collected during the labelling process -> This is a list of tuples -> (phrase, overlap_score)
    """

    # Used for the feature extraction stage - stores the example type for each phrase
    phrase_features = list()
    # Computes the overlap score and the example label
    claims_dict = dict()
    with open(data_file, 'r') as file:
        claims_dict = json.load(file)

    # Using a dictionary here instead of a pandas dataframe prevents formatting issues
    for i in range(0, len(claims_dict.keys())):
        # Evaluates each argument in the current claim
        current_claim = claims_dict[f"{i}"]["claims"]
        for j in range(0, len(current_claim.keys())):
            # This is a string containing the manually labelled target
            labelled_target = current_claim[f"{j}"]["labelled_target"]
            # This is a list containing each noun phrase in the current claim
            candidate_phrases = current_claim[f"{j}"]["candidate_phrase(s)"]

            # Stores the best JI score, phrase and example type for the current best phrase
            best_phrase = [0, "NULL", "NEGATIVE"]
            for k in range(0, len(candidate_phrases)):
                for phrase in candidate_phrases[k]:
                    (JI_score, example_type) = overlap(phrase, labelled_target)
                    # For the feature extraction -> takes the example type for each phrase
                    phrase_features.append((phrase, JI_score))
                    if JI_score > best_phrase[0]:
                        # Updates the best phrase
                        best_phrase[0] = JI_score
                        best_phrase[1] = phrase
                        best_phrase[2] = example_type

            # Once all the candidate phrases have been evaluated, log the relevant labels
            current_claim[f"{j}"]["overlap"] = best_phrase[0]
            current_claim[f"{j}"]["best_phrase"] = best_phrase[1]
            current_claim[f"{j}"]["example_type"] = best_phrase[2]
        
        # Once all the arguments in the given claim have been labelled, update the claim
        claims_dict[f"{i}"] = current_claim
        # This also reduces all claims that have no evidence to an empty set
    
    # Once all the claims have been updated, the data is written to a new JSON file
    with open(output_file, "w") as file:
        json.dump(claims_dict, file, indent=4)

    return phrase_features


def feature_extraction(phrase_features):
    """
    Sorts the phrases by their overlap score and returns the 1000 phrases with the
    highest overlap score, where these are taken to be the features used for the classifier.

    Args:

    Returns:
    """
    # Sorts the features list in place according to their overlap score
    phrase_features.sort(key=lambda x: x[1])
    # Takes the smallest of either the length of the features list or the top 1000 phrases
    end = min(len(phrase_features), 1000)
    # FOR DEBUGGING!
    # print("The features list is:", phrase_features[:end:])
    return phrase_features[:end:]


def claim_features(claim, features):
    """
    Finds the items from the feature set (phrases with the highest overlap) that are
    present in the provided claim.

    Args:
    Returns:
    Returns a dictionary containing the features in the claim
    """
    present_phrases = {}
    for (phrase, JI_score) in features:
        present_phrases[phrase] = (phrase in claim)

    return present_phrases

# Calculate the amount of overlap between the target phrase and the candidate phrase(s)
def classifier(features, labelled_dataset):
    """
    Uses SciPy's logistic regression module to implement an L2 logistic regression classifier
    that works on the json claims data given in the claims_sentences.json file.

    Args:
        feature set
    Returns:
    """
    # Extracts the data from the now labelled dataset
    dataset = dict()
    with open(labelled_dataset, 'r') as file:
        dataset = json.load(file)
    
    # Gets a list of tuples containing each claim and the corresponding example type
    # Necessary for the data to be in this format
    claims = list()
    for claim in dataset.values():
        for j in range(0, len(claim.keys())):
            claims.append((claim[f"{j}"]["text"],
                           1 if claim[f"{j}"]["example_type"] == "POSITIVE" else 0))
    # Converting to binary values of 1 and 0 saves needing to use another library to
    # label the data

    # Seed and randomly suffle the items in the dataset to remove potential biases
    np.random.seed(0)
    np.random.shuffle(claims)
    # Finds the features in each claim, storing this along with the label in a list
    feature_dataset = [(claim_features(claim_text, features), label) for (claim_text, label) in claims]
    # Shuffles and splits the feature dataset into a training and test dataset
    training_data, test_data = train_test_split(feature_dataset, test_size=0.3, random_state=1)

    # Initialises an L2 Regularised Logistic Regression Classification model
    model = SklearnClassifier(LogisticRegression(penalty='l2'))
    # Trains the model on the training data
    model.train(training_data)
    # Finds the accuracy of the model when run on the test data
    accuracy = nltk.classify.accuracy(model, test_data)
    # Output the model's accuracy
    print("The accuracy of the L2-Regularised Logistic regression classifier is:", str(round(accuracy, 2) * 100) + "%")
    # Zips the test data togethet to get the text_features and labels for each feature
    claim_text_features, labels = zip(*test_data)
    # This data is then used to produce a confusion matrix that displays key stats
    print(classification_report(labels,
                               model.classify_many(claim_text_features)))

if __name__ == "__main__":
    # Downloads the English language model (used for syntax parsing)
    stanza.download('en')
    # Reads the JSON case data into a Pandas dataframe (using the caseID as the index)
    df = pd.read_json("data.json", orient='index')
    file_path = './labelled_claims/claim_sentences.json'

    # Experiments/ tests the program using the first 10 items in the dataset
    create_sentence_data(df, file_path)

    # Once the dataset is created/ has been found to have been created
    # It is manually annotated with the claim target
    phrase_features = label_data(file_path,
                                 "./labelled_claims/claim_sentences_LABELLED.json")

    # Execute the L2 Logistic Regression Classifier on the provided dataset and featureset
    # The feature set to be used for this is extracted from the features found above
    classifier(feature_extraction(phrase_features, 0.6),
              "./labelled_claims/claim_sentences_LABELLED.json")

