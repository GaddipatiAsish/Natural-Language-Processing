from nltk.probability import MLEProbDist, FreqDist, LaplaceProbDist, WittenBellProbDist, SimpleGoodTuringProbDist
from nltk.tag import hmm, untag
from nltk.corpus import treebank
from collections import OrderedDict, Counter
from math import log2
import operator

class Q6_Part1(object):

    def load_data(self):
        print("Started Loading the Data")
        # Get the complete data
        data_set = treebank.fileids()
        # Partition the data into train and test data sets
        training_data_fileIds = [file for file in data_set if "wsj_00" in str(file)]
        testing_data_fileIds = [file for file in data_set if "wsj_01" in str(file)]

        tagged_training_data = treebank.tagged_sents(fileids=training_data_fileIds)
        tagged_testing_data = treebank.tagged_sents(fileids=testing_data_fileIds)

        #tagged_training_words = treebank.tagged_words(fileids=training_data_fileIds)
        tagged_testing_words = treebank.tagged_words(fileids=testing_data_fileIds)

        # print(len(tagged_training_data1), len(tagged_testing_data1))

        # UnTag the data for other uses
        untagged_training_data = [untag(item) for item in tagged_training_data]
        untagged_testing_data = [untag(item) for item in tagged_testing_data]

        print("Data Loaded Successfully. Stats are")
        print("Training Data Sentences: ", len(tagged_training_data))
        print("Testing Data  Sentences: ", len(tagged_testing_data))

        return tagged_training_data, tagged_testing_data, tagged_testing_words, untagged_training_data, untagged_testing_data

    def train_hmm(self, training_data, estimator=None):
        """
        "Supervised" Train's a Hidden Markov Model on the training data set
        :param estimator: estimator to smooth the model
        :return: hmm model with given smoothing estimator
        """
        trainer = hmm.HiddenMarkovModelTrainer()
        model = trainer.train_supervised(training_data, estimator)
        print("Training Done! Model Info: ", model)
        return model

    def test_hmm(self, model, test_data_tagged):
        """
        "Supervised" Takes in the model and the dataset to test the model and report the accuracy
        :param model:
        :return:
        """
        model.test(test_sequence=test_data_tagged,verbose=False)

    def find_OOVs(self, train_data, test_data):
        """
        creates of list of OOV words for the given train and test data
        :param train_data: Training data
        :param test_data: Test data
        :return: set of OOV's
        """
        # Create a set of test data words
        test_set = set()
        for item in test_data:
            test_set.update(set(item))

        # Create a set for train data words
        train_set = set()
        for item in train_data:
            train_set.update(set(item))

        # Compute the difference and retain the elements in test set as OOV's
        oovs = test_set.difference(train_set)
        return oovs

    def retreive_tagged_OOVs(self, tagged_data, oovs):
        """
        Takes in tagged test data set and the oov's to retrieve a list of (word, tag) for all oov's
        Here all the (word, tag) tuples will be considered as single sentence
        :param test_data:
        :return:
        """
        tagged_oovs = list()
        temp_oovs = list()
        for sentence in tagged_data:
            for word, tag in sentence:
                if word in oovs:
                    tup = list()
                    tup.append(word)
                    tup.append(tag)
                    temp_oovs.append(tuple(tup))
        tagged_oovs.append(temp_oovs)
        return tagged_oovs

    def remove_words_having_punctations(self, tagged_data):
        """
        Takes in tagged data set as input and removes the words that are having punctuations
        :param tagged data set
        :return tagged data set having alpha numeric words
        """
        sentence_index = range(len(tagged_data))
        indexed_sentences = OrderedDict(zip(sentence_index,tagged_data))
        for sentence_id, sentence in indexed_sentences.items():
            word_index = range(len(tagged_data[sentence_id]))
            indexed_words = OrderedDict(zip(word_index,tagged_data[sentence_id]))
            for word_id, tagged_word in indexed_words.items():
                if not str(tagged_word[0]).isalnum():
                    del indexed_words[word_id]
            indexed_sentences[sentence_id] = list(indexed_words.values())

        updated_tagged_testing_data = list(indexed_sentences.values())
        return updated_tagged_testing_data

    def entropy_of_words(self, tagged_data):
        """
        Takes in tagged words as input and return tag entropies of the words

        """
        tagged_words_fdist = Counter(tagged_data)
        total_no_of_tagged_words = sum(tagged_words_fdist.values())

        untagged_data = untag(tagged_data)
        untagged_words_fdist = Counter(untagged_data)
        total_no_of_words = sum(untagged_words_fdist.values())

        # Create Word Tags dictionary as shown below in format
        word_tags = dict() # {word:{(word,tag1),....(word,tagN)},.....}
        for tagged_word in tagged_words_fdist.keys():
            if tagged_word[0] in word_tags.keys():
                word_tags[tagged_word[0]].add(tagged_word)
            else:
                word_tags[tagged_word[0]] = set()
                word_tags[tagged_word[0]].add(tagged_word)

        # Compute the entropies of the words
        entropies = dict()
        for word in untagged_words_fdist.keys():
            entropies[word] = 0
            tagged_words = word_tags[word]
            for tagged_word in tagged_words:
                yi = tagged_words_fdist[tagged_word]/untagged_words_fdist[word]
                entropies[word] += -(yi*log2(yi))

        entropies = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)
        return entropies, word_tags, tagged_words_fdist

    def update_data_to_contain_top10_words(self, tagged_data, top10):
        """
        Takes in a data set and removes the occurances of all words except top 10 entropy words
        :param tagged_data tagged dataset
        :param top10 top 10 highest entropy words
        :return updated data set
        """
        sentence_index = range(len(tagged_data))
        indexed_sentences = OrderedDict(zip(sentence_index,tagged_data))
        for sentence_id, sentence in indexed_sentences.items():
            word_index = range(len(tagged_data[sentence_id]))
            indexed_words = OrderedDict(zip(word_index,tagged_data[sentence_id]))
            for word_id, tagged_word in indexed_words.items():
                if tagged_word[0] not in top10:
                    del indexed_words[word_id]
            indexed_sentences[sentence_id] = list(indexed_words.values())

        sentence_index = range(len(tagged_data))
        updated_tagged_data = list(indexed_sentences.values())
        indexed_sentences = OrderedDict(zip(sentence_index, updated_tagged_data))
        for sentence_id, sentence in indexed_sentences.items():
            if not sentence:
                del indexed_sentences[sentence_id]
        updated_tagged_data = list(indexed_sentences.values())
        return updated_tagged_data


ob = Q6_Part1()
(x, y, z, a, b) = ob.load_data()

# Test for all words
"""
Logic for cosidering all words as a single sentence
#q = list()
#m = list()
#for sent in y:
#    m = m + sent
#q.append(m)
#print(type(m))
"""


"""
Probability Distribution
"""
#estimator = lambda fdist, bins: MLEProbDist(fdist) # MLE Estimator
#estimator = lambda fdist, bins: LaplaceProbDist(fdist,bins) # Laplace Estimator
#estimator = lambda fdist, bins: WittenBellProbDist(fdist,bins) # Written Bell
estimator = lambda fdist, bins: SimpleGoodTuringProbDist(fdist,bins) # Simple Good Turing
"""
Test For all Words
"""
print("1. For all words")
model = ob.train_hmm(x, estimator)
ob.test_hmm(model, y)

"""
Test for OOV words
"""
print("2. For out-of-vocabulary words")
# Consider all oov's as single sentence
oov_words = ob.find_OOVs(a, b)
# As an oov word might take diff. tag at diff. Instances, make a sentence with (word, tag) tuples for all the oov words
tagged_oov_words = ob.retreive_tagged_OOVs(y, oov_words)
ob.test_hmm(model, tagged_oov_words)

"""
Test for all words with out punctuation
    For Testing:
    - Remove words that are having punctuations in the sentences of test data set.
    - Use the updated list of tagged sentences.
"""
print("3. For all words with out punctuations")
updated_tagged_testing_data = ob.remove_words_having_punctations(y)
ob.test_hmm(model, updated_tagged_testing_data)

"""
Test for Top 10 highest entropy words
"""
tag_entropies_of_words, word_tags, tagged_words_fdist = ob.entropy_of_words(z)
top10_wordentropies = tag_entropies_of_words[:10]
top10words = [word for word, entropy in top10_wordentropies]
print("Top 10 Highest Tag Entropy Words")
for word in top10words:
    tagged_words = word_tags[word]
    print(word, tagged_words)
    x = [tagged_words_fdist[tagged_word] for tagged_word in tagged_words]
    print(x)
updated_tagged_testing_data = ob.update_data_to_contain_top10_words(y, top10words)
print("4. For Top 10 highest entropy words")
ob.test_hmm(model, updated_tagged_testing_data)

