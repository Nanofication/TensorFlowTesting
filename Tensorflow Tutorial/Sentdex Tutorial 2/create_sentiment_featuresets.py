# Get unique words array

# Get a new sentence "I pulled the chair up to the table
# This sentence is converted. This lexicon will be np.zeros(len(lexicon))
# Then you check does I exist in lexicon? no = 0. Yes? = 1

# Example:
# Lexicon array [chair, table, spoon, television]
# Sentence: "I pulled the chair up to the table
# NN array [1,1,0,0] Chair is index 1 table is index 2. This is now our vector.

# You should run this 1 time and never do this again. Always use from pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # Running, Ran, Run => These are all Run

import numpy as np
import random
import pickle # We need to save this data as some point
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

# If you get a MemoryError <-- You ran out of ram or vram
# Our model will be put into ram here.
# The larger your dataset, the more ram you will take up
# The main value in a neural network is to feed it an insane amount of data

def create_lexicon(pos,neg):
    lexicon = []
    for file in [pos,neg]:
        with open(file,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon) # This gives us a dictionary like element
    # w_counts = {'the': 52521 (occurs), 'and':25242}

    l2 = []
    for w in w_counts:
        if 1000> w_counts[w] > 50:
            l2.append(w)
    # We don't want super common words. In future, we will want words that matter like "Email"
    # Ideally we want lexicon to be as small as possible
    # This part does not matter as greatly

    print(len(l2))
    return l2
    # Do we care about every single word? Or a range of words?
    # Lemmatize first


def sample_handling(sample, lexicon, classification):
    featureset = []

    """
    featureset = [
        [[0 1 0 1 1 0], [1, 0]], # This means a list of lists where within the lists
        # you will have this "hot" array with labels that classifies what sentence it is
        # In this example. [1,0] is positive sentence sample or [0,1] negative sentence sample
    ]
    """

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon)) # Initialize zeros in array

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower()) # Search in our lexicon for the lower case word
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])

    random.shuffle(features)
    # Purpose of shuffle Does tf.argmax([output]) == tf.argmax([expectations])?
    # Did we get the prediction right?
    # If not shuffle, you train the network to just keep getting positive. Data will be very biased

    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    # examples [[5,8],[7,9]] :,0 means I want all the zero elements. These are our actual features
    # Datas are all structured like this [features, labels]

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)