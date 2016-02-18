"""
Author: Gaddipati, Asish kumar
Ex3_part2.py preprocess the train and test data
    1. creates vocabulary that appears atleast twice in the training data
    2. Edit the training corpus by replacing all hapax legomena's with <unk> token and numbers with <num> token
    3. Edit the test corpus by replacing oov words with <unk> and numbers with <num>
and then pickles the updated training data for future use.

Run this program using python3
Output: 
	create two pickled files train_data.pkl and test_data.pkl of pre processed data.
"""
# Import the necessary packages
from nltk.corpus import brown
from nltk.probability import FreqDist
import re
import pickle

# Choose the First file in each category for testing and rest for training
test_file_ids = list() # List of test files
train_file_ids = list()# List of train files
for category in brown.categories():
    test_file_ids.append(brown.fileids(category)[0])
    train_file_ids.extend(brown.fileids(category)[1:])

# Tokenize the raw data into words for training as well as testing
test_data = list()
train_data = list()
for fid in test_file_ids: # Make sure language models are case insensitive
    test_data.extend([word.lower() for word in brown.words(fileids=fid)])
#print(len(test_data))

for fid in train_file_ids:
    train_data.extend([word.lower() for word in brown.words(fileids=fid)])
#print(len(train_data))

"""Preprocessing the train data set"""
print("Train, Tokens: Before Preprocessing", len(train_data))
# 1. Create a Vocabulary contain words that appear atleast twice
fdist = FreqDist(train_data) # Compute the freq dist of training data
hapax = fdist.hapaxes() # Find the hapaxes in Training data
train_data_vocab = set(fdist.keys()) # Find the total Vocabulary of training set

for word in hapax: # Update the Vocabulary to contain words that occurred at least twice
    if word in train_data_vocab:
        train_data_vocab.remove(word)
# 2. Update hapax legomena with <unk>
print("Updating the train data with <unk>.....It takes time!wait")
count = 0
for word in hapax:# Replace hapaxes with <unk> in training data
    if word in train_data:
        count = count + 1
        index = train_data.index(word)
        train_data.remove(word)
        train_data.insert(index,"<unk>")
print("No of Train Tokens updated with <unk> are ", count)
# 3. Update numbers with <num>
count = 0
for num_string in train_data: # Update numbers in train data set to <num>
    if len(re.findall(r"[+-]?\d+?(?:\.\d+)?(?:[eE][+-]?\d+)?", num_string)) > 0:
        count = count + 1
        index = train_data.index(num_string)
        train_data.remove(num_string)
        train_data.insert(index,"<num>")
print("No of Train Tokens updated with <num> are ", count)
print("Train, Tokens: After Preprocessing", len(train_data))

# 4. Get the updated Vocabulary that having <num>, <unk> and words that happened atleast twice in train dataset
train_data_vocab = set(train_data)

# 5. Pickle the train data into files
fd1 = open("train_data.pkl",'wb')
pickle.dump((train_data,train_data_vocab), fd1, protocol=2)


"""PreProcessing the Test data set"""

print("Test, Tokens: Before Preprocessing", len(test_data))
# 1. Remove the out of vocab words from test corpus
test_data_vocab = set(test_data) # Find the vocabulary of the test data
oov_words = test_data_vocab.difference(train_data_vocab) # Find the out of vocabulary words
print("updating the test data....It takes time! wait")
# 2. Update numbers with <num>
count = 0
for num_string in test_data:
    if len(re.findall(r"[+-]?\d+?(?:\.\d+)?(?:[eE][+-]?\d+)?", num_string)) > 0:
        count = count + 1
        index = test_data.index(num_string)
        test_data.remove(num_string)
        test_data.insert(index,"<num>")
print("No of Test Tokens updated with <num> are ", count)
# 3. Update the oov words of test data with <unk>
count = 0
print(oov_words)
for word in oov_words:
    if word in test_data:
        count = count + 1
        index = test_data.index(word)
        test_data.remove(word)
        test_data.insert(index,"<unk>")
print("No of Test Tokens updated with <unk> are ", count)
print("Test, Tokens: After Preprocessing", len(test_data))
# 4. Pickle the test data to a file
fd2 = open("test_data.pkl",'wb')
pickle.dump((test_data,test_data_vocab,oov_words), fd2, protocol=2)

