# Get unique words array

# Get a new sentence "I pulled the chair up to the table
# This sentence is converted. This lexicon will be np.zeros(len(lexicon))
# Then you check does I exist in lexicon? no = 0. Yes? = 1

# Example:
# Lexicon array [chair, table, spoon, television]
# Sentence: "I pulled the chair up to the table
# NN array [1,1,0,0] Chair is index 1 table is index 2. This is now our vector.

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

