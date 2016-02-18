"""

Author: Gaddipati, Asish Kumar
Ex4.py tries to solve authorship attribution task using two approaches
for Approach 1: 
    1. Create a training data set having <unk> for hapax legomena
    2. Build the trigram language models
    3. Tabulate the perplexities
For Approach 2:
    1. Build Unigram model for each author and compute the entropies for each author
	on these models.

Run this program using python2
 
"""
# Import the necessary packages
from nltk.corpus import gutenberg
from nltk.probability import FreqDist, LidstoneProbDist
from nltk.tokenize import word_tokenize
from nltk.model import NgramModel

class Ex4(object):
    """
	All the necessary variables and methods to implement the Author Attribution task is
	implemented in this class. 
    """
    # Class variables to store the raw form of train and test data sets
    # for Austen
    AustenTrain = ""
    AustenTest = ""
    # For Chesterton
    ChestertonTrain = ""
    ChestertonTest =  ""
    # For Shakespeare
    ShakespeareTrain= ""
    ShakespeareTest= ""
    # Bible data
    BibleTest = ""

    def __init__(self):
        """
        Reads the raw train/test data from the filesystem and loads it into class variables
        """
        # Read the Training and Test data and convert to lower case
        # Read the Austen data for training and testing
        self.AustenTrain = gutenberg.raw('austen-persuasion.txt').lower()
        self.AustenTrain += gutenberg.raw('austen-sense.txt').lower()

        self.AustenTest = gutenberg.raw('austen-emma.txt').lower()

        # Read the Chesterton data for training and testing
        self.ChestertonTrain = gutenberg.raw('chesterton-thursday.txt').lower()
        self.ChestertonTrain += gutenberg.raw('chesterton-brown.txt').lower()

        self.ChestertonTest = gutenberg.raw('chesterton-ball.txt').lower()

        # Read the Shakespeare data for training and testing
        self.ShakespeareTrain = gutenberg.raw('shakespeare-macbeth.txt').lower()
        self.ShakespeareTrain += gutenberg.raw('shakespeare-caesar.txt').lower()

        self.ShakespeareTest = gutenberg.raw('shakespeare-hamlet.txt').lower()

        # Read the bible data for test
        self.BibleTest = gutenberg.raw('bible-kjv.txt').lower()

    def preprocess_data(self, train_data):
        """
        preprocess the train data by replacing hapaxes with <unk>'s
        :param train_data: word tokenized train data
        :return: updated train data with <unk>'s
        """
        count = 0
        for word in FreqDist(train_data).hapaxes():
            if word in train_data: # Update the training data with <unk>
                count = count + 1
                index = train_data.index(word)
                train_data.remove(word)
                train_data.insert(index,"<unk>")
        print("No of Train Tokens updated with <unk> are ", count)
        return train_data

    def build_language_model(self, n, train_data, gamma):
        """
        builds the language model for a given n_gram and training data
        :param n: n_gram
        :param train_data: word tokenized train data
        :return: language mode
        """
        est = lambda fdist, bins: LidstoneProbDist(fdist, gamma)
        return NgramModel(n, train_data, estimator=est)

    def authorship_test(self,lm_austen,lm_chesterton,lm_shakespeare, test_data):
        """
        computes the perplexity of the test data on the 3 trained language models
        :param lm_austen: language model from austen training data
        :param lm_chesterton: language model from chesterton training data
        :param lm_shakespeare: language model from shakespeare training data
        :param test_data: word tokenized test data
        :return: list of perplexities
        """
        prplx_on_austen_lm = lm_austen.perplexity(test_data)
        prplx_on_chesterton_lm = lm_chesterton.perplexity(test_data)
        prplx_on_shakespeare_lm = lm_shakespeare.perplexity(test_data)
        return (prplx_on_austen_lm, prplx_on_chesterton_lm, prplx_on_shakespeare_lm)

    def authorship_test_stons(self,lm_austen,lm_chesterton,lm_shakespeare, test_data):
        
        """
	Computes the entropy of the test data having singleton's on given language models
        :param lm_austen: language model from austen training data
        :param lm_chesterton: language model from chesterton training data
        :param lm_shakespeare: language model from shakespeare training data
        :param test_data: word tokenized test data
        :return: list of entropy values.
	"""
        
        entr_on_austen_lm = lm_austen.entropy(test_data)
        entr_on_chesterton_lm = lm_chesterton.entropy(test_data)
        entr_on_shakespeare_lm = lm_shakespeare.entropy(test_data)
        return (entr_on_austen_lm, entr_on_chesterton_lm, entr_on_shakespeare_lm)
	"""
	austen_gm = 1
        chesterton_gm =1
        shakespeare_gm = 1
        n = len(test_data)
        for word in test_data:
            austen_gm *= lm_austen.prob(word,word);
            chesterton_gm *= lm_chesterton.prob(word,word)
            shakespeare_gm *= lm_shakespeare.prob(word,word)
        return(pow(austen_gm,1/n), pow(chesterton_gm,1/n), pow(shakespeare_gm, 1/n))
    	"""
# Instantiate the class
ex4_part1 = Ex4()

""" Experiment 1"""

# Update the training data not to contain hapaxes with <unk>
austen_train = ex4_part1.preprocess_data(word_tokenize(ex4_part1.AustenTrain))
chesterton_train = ex4_part1.preprocess_data(word_tokenize(ex4_part1.ChestertonTrain))
shakespeare_train = ex4_part1.preprocess_data(word_tokenize(ex4_part1.ShakespeareTrain))

# Get the language model using pre processed data
lm_austen = ex4_part1.build_language_model(3, austen_train, 0.2)
lm_chesterton = ex4_part1.build_language_model(3, chesterton_train, 0.2)
lm_shakespeare = ex4_part1.build_language_model(3, shakespeare_train, 0.2)

# Print the perplexity values
print ex4_part1.authorship_test(lm_austen,lm_chesterton,lm_shakespeare, word_tokenize(ex4_part1.AustenTest))
print ex4_part1.authorship_test(lm_austen,lm_chesterton,lm_shakespeare, word_tokenize(ex4_part1.ChestertonTest))
print ex4_part1.authorship_test(lm_austen,lm_chesterton,lm_shakespeare, word_tokenize(ex4_part1.ShakespeareTest))
print ex4_part1.authorship_test(lm_austen,lm_chesterton,lm_shakespeare, word_tokenize(ex4_part1.BibleTest))

""" Experiment 2 """
# Get the Unigram language model using original train data
lm_austen = ex4_part1.build_language_model(1, word_tokenize(ex4_part1.AustenTrain), 0.0001)
lm_chesterton = ex4_part1.build_language_model(1, word_tokenize(ex4_part1.ChestertonTrain),0.0001)
lm_shakespeare = ex4_part1.build_language_model(1, word_tokenize(ex4_part1.ShakespeareTrain), 0.0001)

# Print the Entropies of the test data on 3 language models
print ex4_part1.authorship_test_stons(lm_austen,lm_chesterton,lm_shakespeare, FreqDist(word_tokenize(ex4_part1.AustenTest)).hapaxes())
print ex4_part1.authorship_test_stons(lm_austen,lm_chesterton,lm_shakespeare, FreqDist(word_tokenize(ex4_part1.ChestertonTest)).hapaxes())
print ex4_part1.authorship_test_stons(lm_austen,lm_chesterton,lm_shakespeare, FreqDist(word_tokenize(ex4_part1.ShakespeareTest)).hapaxes())
print ex4_part1.authorship_test_stons(lm_austen,lm_chesterton,lm_shakespeare, FreqDist(word_tokenize(ex4_part1.BibleTest)).hapaxes())

#print ex4_part1.authorship_test_stons(austen_lid,chesterton_lid,shakespeare_lid, FreqDist(word_tokenize(ex4_part1.AustenTest)).hapaxes())
#print ex4_part1.authorship_test_stons(austen_lid,chesterton_lid,shakespeare_lid, FreqDist(word_tokenize(ex4_part1.ChestertonTest)).hapaxes())
#print ex4_part1.authorship_test_stons(austen_lid,chesterton_lid,shakespeare_lid, FreqDist(word_tokenize(ex4_part1.ShakespeareTest)).hapaxes())
#print ex4_part1.authorship_test_stons(austen_lid,chesterton_lid,shakespeare_lid, FreqDist(word_tokenize(ex4_part1.BibleTest)).hapaxes())
