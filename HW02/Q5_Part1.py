from nltk.probability import FreqDist
from nltk.corpus import brown, treebank
from nltk.tag import untag
from collections import OrderedDict
import pickle as pkl



class Q5_Part1(object):
    """
    Q5_Part1 preprrocess the data set and pickles the file for future use.
    """

    def load_data(self, data_set):
        """
        Loads the given data set. Makes the data set case insensitive.
        Remove words that appear less than 5 times.
        :return updated data set
        """
        print("Started Loading the Data")
        tagged_tokens = data_set.tagged_words()
        tokens = untag(tagged_tokens)

        # Get the list of words that appear less than 5 times in  Corpus
        print("Get LT5's")
        tokens = [token.lower() for token in tokens] # Convert to lower case
        freq_dist = FreqDist(tokens) # Compute the freq dist
        tokens_lt_5 = [word for word, count in freq_dist.items() if count < 5]

        # Delete words less than 5 and make the corpus insensitive
        print("Making data case insensitive")
        token_range = range(len(tagged_tokens))
        indexed_tokens = OrderedDict(zip(token_range,tagged_tokens))
        updated_tagged_tokens = OrderedDict()
        for tagged_token_id, tagged_token in indexed_tokens.items():
            if tagged_token[0].lower() in tokens_lt_5:
                del indexed_tokens[tagged_token_id]
            else:
                temp = list()
                temp.append(tagged_token[0].lower())
                temp.append(tagged_token[1])
                temp = tuple(temp)
                updated_tagged_tokens[tagged_token_id] = temp
        tagged_tokens = list(updated_tagged_tokens.values())

        # Pickle the data for future purpose
        print("Pickling the Updated Corpus")
        if data_set == brown:
            file_name = "q5_brown_updated.pkl"
        else:
            file_name = "q5_treebank_updated.pkl"
        pkl.dump((tagged_tokens, tokens_lt_5), open(file_name,'wb'))

        return tagged_tokens, tokens_lt_5

ob = Q5_Part1()
"""
USING BROWN CORPUS
"""
data_set1 = brown
data_set2 = treebank
"""
Load the data
"""
# Load the data
tagged_tokens, tokens_lt_5 = ob.load_data(data_set1)
tagged_tokens, tokens_lt_5 = ob.load_data(data_set2)