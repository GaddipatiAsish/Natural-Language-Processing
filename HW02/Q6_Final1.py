from nltk.probability import (MLEProbDist, FreqDist, LaplaceProbDist,
                              WittenBellProbDist, SimpleGoodTuringProbDist, ConditionalFreqDist, ConditionalProbDist)
from nltk.tag import hmm, untag
from nltk.corpus import treebank
from collections import OrderedDict, Counter
from math import log2
import operator

class Q6_Final1(object):

    def load_data(self, percentage):
        print("Started Loading the Data")
        # Get the list of file available in penn tree bank
        data_set = treebank.fileids()
        # Partition the data into train and test data sets
        training_data_fileIds = [file for file in data_set if "wsj_00" in str(file)]
        testing_data_fileIds = [file for file in data_set if "wsj_01" in str(file)]

        # How much percentage of files consider for training?
        index = int(percentage*len(training_data_fileIds))
        training_data_fileIds = training_data_fileIds[:index]

        tagged_training_sents = treebank.tagged_sents(fileids=training_data_fileIds)
        tagged_testing_sents = treebank.tagged_sents(fileids=testing_data_fileIds)

        tagged_training_words = treebank.tagged_words(fileids=training_data_fileIds)
        tagged_testing_words = treebank.tagged_words(fileids=testing_data_fileIds)

        # UnTag the data for other uses
        untagged_training_sents = [untag(item) for item in tagged_training_sents]
        untagged_testing_sents = [untag(item) for item in tagged_testing_sents]

        print("Data Loaded Successfully. Stats are")
        print("Training Data Sentences: ", len(tagged_training_sents))
        print("Testing Data  Sentences: ", len(tagged_testing_sents))

        return tagged_training_sents, tagged_testing_sents, tagged_training_words, tagged_testing_words, untagged_training_sents, untagged_testing_sents

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
        "Supervised" Takes in the model and the tagged data set "SENTS" to test the model and report the accuracy
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

    def acc(self, special_words, dataset_actual_tags, dataset_predicted_tags):
        """
        acc method computes the accuracy of the given actual tagged data and the predicted data set considering only the
        specialwords set
        :param special_words that are to be considered for caliculating the accuracy
        :param dataset_actual_tags actual test data set with tags
        :param dataset_predicted_tags test data set having predicted tags
        :return accuracy
        """
        if len(dataset_actual_tags) == len(dataset_predicted_tags):
            numerator = 0; denominator = 0
            sentence_index = range(len(dataset_actual_tags))
            indexed_sentences_actual = OrderedDict(zip(sentence_index,dataset_actual_tags))
            indexed_sentences_predicted = OrderedDict(zip(sentence_index,dataset_predicted_tags))
            for sentence_id in sentence_index:
                actual_tagged_sent = indexed_sentences_actual[sentence_id]
                predicted_tagged_sent = indexed_sentences_predicted[sentence_id]
                if len(actual_tagged_sent) == len(predicted_tagged_sent):
                    for word_id in range(len(actual_tagged_sent)):
                        tagged_word_actual = actual_tagged_sent[word_id]
                        tagged_word_predicted = predicted_tagged_sent[word_id]
                        if tagged_word_actual[0] in special_words:
                            denominator += 1
                            if tagged_word_actual[1] == tagged_word_predicted[1]:
                                numerator += 1
        print("numerator: ", numerator, "denominator: ", denominator)
        accuracy = numerator/denominator*100
        print("accuracy: ", accuracy)
        return accuracy

    def acc_all(self, dataset_actual_tags, dataset_predicted_tags):
        """
        :param actual: actual tags of the dataset
        :param predicted: predicted tags for each word using MOST LIkely Tagger
        :return: returns the accuracy
        """
        if len(dataset_actual_tags) == len(dataset_predicted_tags):
            numerator = 0; denominator = 0
            sentence_index = range(len(dataset_actual_tags))
            indexed_sentences_actual = OrderedDict(zip(sentence_index, dataset_actual_tags))
            indexed_sentences_predicted = OrderedDict(zip(sentence_index, dataset_predicted_tags))
            for sentence_id in sentence_index:
                actual_tagged_sent = indexed_sentences_actual[sentence_id]
                predicted_tagged_sent = indexed_sentences_predicted[sentence_id]
                if len(actual_tagged_sent) == len(predicted_tagged_sent):
                    for word_id in range(len(actual_tagged_sent)):
                        tagged_word_actual = actual_tagged_sent[word_id]
                        tagged_word_predicted = predicted_tagged_sent[word_id]
                        denominator += 1
                        if tagged_word_actual[1] == tagged_word_predicted[1]:
                            numerator += 1
        print("Numerator:", numerator, "Denomenator: ", denominator)
        accuracy = numerator/denominator*100
        return accuracy

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

    def most_likely_tagger_train(self, train_data, smoother=MLEProbDist):
        """
        Takes in the tagged data set in the form of words and return the conditional probability distribution
        with condition based on word
        :param train_data tagged training data
        :return model
        """
        model = ConditionalFreqDist(train_data)
        return model

    def most_likely_tagger_test(self, oovs , test_data, model):
        """
        Takes in the conditional freq distribution as model and labelled test data in the form of sentences
        For oov words the model assigns NN
        :param test_data labelled test data in the form of list of sentences
        :param oovs to assign NN tag
        :return accuracy
        """
        predicted = list()
        for sentence in test_data:
            predicted_sentence = list()
            for tagged_word in sentence:
                predicted_word = list()
                predicted_word.append(tagged_word[0])
                if tagged_word[0] in oovs:
                    predicted_word.append("NN")
                else:
                    predicted_word.append(model[tagged_word[0]].max())
                predicted_word = tuple(predicted_word)
                predicted_sentence.append(predicted_word)
            predicted.append(predicted_sentence)
        return predicted

ob = Q6_Final1()
(x, y, w, z, a, b) = ob.load_data(percentage=1)

"""
Probability Distribution
"""
estimator = lambda fdist, bins: MLEProbDist(fdist) # MLE Estimator
estimator = lambda fdist, bins: LaplaceProbDist(fdist,bins) # Laplace Estimator
estimator = lambda fdist, bins: WittenBellProbDist(fdist,bins) # Written Bell
estimator = lambda fdist, bins: SimpleGoodTuringProbDist(fdist,bins) # Simple Good Turing
"""
Train the Model
"""
model = ob.train_hmm(x, estimator)
"""
1. Accuracy of all the words
"""
print("1. Accuracy of all the words")
ob.test_hmm(model, y)
"""
2. Accuracy for OOV's
"""
print("2. Accuracy for OOV's")
oovs = ob.find_OOVs(a, b)
predicted_tags = [model.tag(sent) for sent in b]
accuracy = ob.acc(oovs, y, predicted_tags)
"""
3. Accuracy for all words with out punctuations
    For Testing:
    - Remove words that are having punctuations in the sentences of test data set.
    - Use the updated list of tagged sentences.
"""
#print("3. For all words with out punctuations")
#updated_tagged_testing_data = ob.remove_words_having_punctations(y)
#ob.test_hmm(model, updated_tagged_testing_data)
# Find the words without having punctuations
print("Accuracy for all words with out punctuations")
words_without_punctuation = set()
for sentence in y:
    for tagged_word in sentence:
        if str(tagged_word[0]).isalnum():
            words_without_punctuation.add(tagged_word[0])
ob.acc(words_without_punctuation, y, predicted_tags)
"""
4. Accuracy for High Entropy words
"""
print("4. Accuracy for High Entropy words")
tag_entropies_of_words, word_tags, tagged_words_fdist = ob.entropy_of_words(z)
top10_wordentropies = tag_entropies_of_words[:10]
top10words = [word for word, entropy in top10_wordentropies]
print("Top 10 Highest Tag Entropy Words")
for word in top10words:
    tagged_words = word_tags[word]
    print(word, tagged_words)
    counts = [tagged_words_fdist[tagged_word] for tagged_word in tagged_words]
    print(counts)
ob.acc(top10words, y, predicted_tags)

"""
Most Likely Tag Tagger
"""
model = ob.most_likely_tagger_train(w)
oovs = ob.find_OOVs(w,z)
predicted_tags = ob.most_likely_tagger_test(oovs, y, model)
# Accuracy for all the words
print("Accuracy for all words : ", ob.acc_all(y, predicted_tags))
# Accuracy for OOVS
print("Accuracy for OOV's :", ob.acc(oovs,y,predicted_tags))
# Accuracy for words with out punctuations
words_without_punctuation = set()
for sentence in y:
    for tagged_word in sentence:
        if str(tagged_word[0]).isalnum():
            words_without_punctuation.add(tagged_word[0])
print("Accuracy for words with out punctuations", ob.acc(words_without_punctuation, y, predicted_tags))
# Accuracy for Top 10 Highest Entropy Words
tag_entropies_of_words, word_tags, tagged_words_fdist = ob.entropy_of_words(z)
top10_wordentropies = tag_entropies_of_words[:10]
top10words = [word for word, entropy in top10_wordentropies]
print("Top 10 Highest Tag Entropy Words")
for word in top10words:
    tagged_words = word_tags[word]
    print(word, tagged_words)
    counts = [tagged_words_fdist[tagged_word] for tagged_word in tagged_words]
    print(counts)
print("Accuracy for Top 10 Highest Entropy Words :", ob.acc(top10words, y, predicted_tags))