import numpy as np
import time
import string
import stanza
import nltk
from nltk.tokenize import word_tokenize
from stanza.models.constituency.parse_tree import Tree
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


# Downloads the English language model
stanza.download('en')

# Initialises the English neural network pipeline
nlp_pipeline = stanza.Pipeline('en', processors='tokenize,mwt,pos,constituency')
nlp_object = nlp_pipeline("The quick brown fox jumps over the lazy dog. What happens when this becomes a much larger piece of text? Say for example when the defendant/ prosecutor is trying to make a well thought out case?")

for sentence in nlp_object.sentences:
    tree = sentence.constituency
    # Tests the breadth first traversal algorithm
    NPs = breadth_first_search(tree, "NP")

    print("The Noun Phrases are:", NPs)

tree = nlp_object.sentences[0].constituency
# print("NOUN PHRASES:", breadth_first_search(tree, "NP"))

print("Set of the given phrase is:")
sent = "The quick brown fox jumps over the lazy dog. What happens when this becomes a much larger piece of text? Say for example when the defendant/ prosecutor is trying to make a well thought out case?"

# Remove punctuation and convert the sentence into lowercase
# sent = sent.lower()
sent = ''.join(char for char in sent if char not in string.punctuation).lower()
# Performing spelling correction for each case is a computationally expensive task
# That is not done here to save time (likely to have a marginal affect anyhow)
print(set(word_tokenize(sent)))


print("---------------------------------------------------------------------------------")
print("Testing how to tokenise sets")
sentence_un = "This is a setence containing text."
sentence_deux = "This is yet another sentence containing some text."

sentence_un_set = set(word_tokenize(sentence_un))
print(sentence_un_set)
sentence_deux_set = set(sentence_deux)
print(sentence_deux_set)
