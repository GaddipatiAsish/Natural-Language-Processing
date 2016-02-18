"""
Q5. Language Detection
Given a piece of unknown text, the task is to find what language it belongs to?
Author: Gaddipati, Asish Kumar

Run this program with python2

Output:
Confusion Matrix showing the accuracy %'s of language detection tasks.
"""

# Import all the required packages
import re
from nltk.corpus import europarl_raw
from nltk.tokenize import sent_tokenize
from nltk.probability import MLEProbDist
from nltk.probability import LidstoneProbDist
from nltk.model.ngram import NgramModel
import sys


class Ex5(object):
    """ all the methods needed for Langugage detection task are defined in Q5"""
    # class variables
    languages = [europarl_raw.english, europarl_raw.danish, europarl_raw.dutch ,europarl_raw.finnish ,europarl_raw.french,
                 europarl_raw.german, europarl_raw.greek, europarl_raw.italian, europarl_raw.portuguese, europarl_raw.spanish,
                 europarl_raw.swedish]  # list of languages considered for training and testing
    #vocabs = []
    #languages = [europarl_raw.english, europarl_raw.danish]   
    def __init__(self):
        # Make the system default to utf-8 to deal with special charactors in all languages
        reload(sys)
        sys.setdefaultencoding('utf-8')

    def make_dataset(self, lang_dir, dataset_type, No_files):
        """
        Makes the train and test data sets for a given language
        :param language: specify the language for which a data set has to be made
               dataset_type : specify if it is a train or test data set
               No_files: specify the no of original data set files to use for test dataset
        :return: train or test data set
        """
        # Get the list of files in the given language folder
        listOfFiles = lang_dir.fileids()

        sentences = list() # Instantiate a list of sentences
        # print len(listOfFiles)
        # count = 0
        
        # Choose What data set to make
        if dataset_type is "train":
           files = listOfFiles[0:len(listOfFiles)-No_files]
        else:
            files = listOfFiles[len(listOfFiles)-No_files : len(listOfFiles)]
        # Start preprocessing the document
        for file in files:
            print("Processing Document : ", file)
            temp_sents = lang_dir.sents(file) # Read the tokenized sentences
            # count = count + len(enTrainSents)
            for sentence in temp_sents:
                sentence_lower = [word.lower() for word in sentence]# Convert into a lower case sentence
                #print type(sentence1)
                for i in range(0,len(sentence_lower)):
                    temp = ""
                    for letter in sentence_lower[i]:# Removing the numbers from the sentence
                        if not letter.isnumeric():
                            temp += letter
                    temp1 = re.sub(' +','',temp)
                    sentence_lower[i] = temp1
                #sentence_lower = re.sub(' +','', sentence_lower)# Remove multiple spaces by single space if any
                sentences.append(sentence_lower) # Final PreProcessed Sentences
	#print count
	#print enTrainSentences[0]
	#print enTrainSentences[500]

        # Tokenize the pre processed sentences into charactors
        charactors = list()
        if dataset_type is "train": # In case of Training, just return the list of charactors for the complete training data
            for sentence in sentences:
                for word in sentence:
                    for char_index in range(len(word)):
                        charactors.append(word[char_index])
        else:
            for sentence in sentences:# In case of testing, return as list of lists [[],[],[]]
                temp_s =list()
                for word in sentence:
                    temp_w = list()
                    temp_w = [word[char_index] for char_index in range(len(word))]
                    temp_s.append(temp_w)
                charactors.append(temp_s)	
	
	#print len(charactors)
	#print charactors[5]
	#print len(set(charactors))
        return charactors


    def build_language_model(self, train_charactors):
        """ Builds the Bigram Language Model for a given Training data set using Lidstone Estimation"""

        est = lambda fdist, bins: LidstoneProbDist(fdist,bins) # Lidstone Estimator
        lm = NgramModel(2, train_charactors, estimator = est) # Bigram Model
        #print "Inside Language Model"
	#print len(set(train_charactors))
	#print len(lm._ngrams)
	return lm #Return the build model
 
    def compute_match_percentage(self, trained_lms, vocabularies, test_data):
        counts_of_matches = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
        #counts_of_matches = {0:0,1:0}
	for sentence in test_data:
	    #print "Sentence is",sentence
	    #print len(trained_lms)
            #print len(vocabularies)
            sent = list()
            for word in sentence: # Convert the sentence into a list of words
                for letter in word: # For each letter of the word
		    sent.append(letter)
	    #print sent
            i = 0
            temp_counts = list();
            for lm in trained_lms: # Compute perplexity of that sentence for each language model
                #print "Nothing Happens here"
		#print set(sent).issubset(vocabularies[i])
		if set(sent).issubset(vocabularies[i]) is False:
                    #print set(sent),"not  a subset of ",vocabularies[i]
		    temp_counts.append(1000000)
		    #print "Infinite Perplexity Reported"
                elif set(sent).issubset(vocabularies[i]) is True:
		    #print "computing perplexity"
                    temp_counts.append(lm.perplexity(sent))
                i = i + 1
            ##print(temp_counts)
	    model = temp_counts.index(min(temp_counts)) # find the model with min_perplexity
	    ##print "Model chose is: ", model
	    counts_of_matches[model] = counts_of_matches.get(model) + 1 
        print counts_of_matches	
	print "Total Sentences Reported", 
	tot_sents = sum(counts_of_matches.values())
	print "Total Sentences Reported", tot_sents
	#print [value/tot_sents*100 
	#for key,value in counts_of_matches.items():
	#    print((value/tot_sents)*100)

ob = Ex5() # Instantiate the class 
lms = list() # List of language models
vocabs = list() # Trainig Vocabularies of different languages
for language in ob.languages:# For each Language
    print(language)
    train = ob.make_dataset(language, "train", 1);# Pre process the data set
    #test = ob.make_dataset(language, "test", 1)
    vocabs.append(set(train)) # Make a vocabulary
    lm = ob.build_language_model(train) # Build the language model
    #print(lm)
    #lm.perplexity(test)
    lms.append(lm) # Append the language model to list lms

k = 0
for language in ob.languages: # For each language 
    print(language)
    test = ob.make_dataset(language, "test", 1) # make the data set for testing 
    print "Total No of Sentences in Test",len(test)
    ob.compute_match_percentage(lms, vocabs, test) #Compute the match percentage
    k = k+1
