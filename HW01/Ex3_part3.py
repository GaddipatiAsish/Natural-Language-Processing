"""
Author: Asish Kumar Gaddipati
Ex1_part3.py tries to solve Q3.e by doing below
    1. import the pickled data from Ex1_part2.py
    2. Build the language models
    3. Solve (e) of Q3
Run this program using python2

"""
# Import the necessary packages
import pickle
from nltk.probability import LidstoneProbDist
from nltk.model import NgramModel
import numpy as np
import random

# Import the pickled data
fd = open("train_data.pkl",'rb')
(train_data,train_data_vocab) = pickle.load(fd)
fd = open("test_data.pkl",'rb')
(test_data,test_data_vocab,oov_words) = pickle.load(fd)

# Use Listone with gamma = 0.2 as estimator
est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)

# Build all the three models
lm1 = NgramModel(1, train_data, estimator = est)
lm2 = NgramModel(2, train_data, estimator = est)
lm3 = NgramModel(3, train_data, estimator = est)

lms = (lm1, lm2, lm3)

# Logic for mixxing is shown here
mixes = [0,0.0001,0.001,0.01,0.05,0.1];

for lm in lms: # For each language
    print lm
    for mix in mixes: # For each Mix value
        for i in range(0,10):# Repeat the Experiment 10 times for each mix
            arr = np.empty([10, 1])
	    for word in test_data:# Update the training data for a given mix
	        ra = random.random() # Generate a random number between 0 and 1
	        if ra > mix: # liklihood check
	           rndm_word = lm.choose_random_word(test_data_vocab)
	           ind_word = test_data.index(word)
	           test_data.pop(ind_word)
	           test_data.insert(ind_word,rndm_word)
            arr[i] = lm.perplexity(test_data) # Compute the perplexity
        print "Mean perplex is ", np.mean(arr), "for mix ", mix # Compute the mean perplexity
        print "STD is "	, np.std(arr), "for mix ", mix # Compute the standard deviation

