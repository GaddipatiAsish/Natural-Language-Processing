from nltk.tag import untag
from collections import Counter
import pickle as pkl
from math import log2
import operator

class Q5_Part2(object):
    """
    Q5_Part2 makes a lists of words based upon the no of tags possibly it can take.
    It also computes the entropies of the words and prints the top 10 highest entropy words of the given data set
    """
    def load_pickled_data(self):
        """
        Load the pickled data set and words that appeared less than 5 times into memory
        :return:
        """
        tagged_data, tokens_lt_5 = pkl.load(open("q5_brown_updated.pkl",'rb'))
        return tagged_data, tokens_lt_5

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

        return entropies, word_tags, untagged_words_fdist

ob = Q5_Part2()
"""
Load the data
"""
tagged_data, tokens_lt_5 = ob.load_pickled_data()
"""
Compute the Entropies [(word, count, {tags}, entropy),......]
"""
tag_entropies_of_words, word_tags, untagged_words_fdist = ob.entropy_of_words(tagged_data)
top10_wordentropies = sorted(tag_entropies_of_words.items(), key=operator.itemgetter(1), reverse=True)
top10_wordentropies = top10_wordentropies[:10]
print("Printing the Top 10 words")
for word, entropy in top10_wordentropies:
    print(word, untagged_words_fdist[word], word_tags[word], entropy)

"""
Compute the lists and pickle them
"""
# Make a dictionary as shown below
#{no_of_tags:(words),.......}
tag_words_dict = dict()
for word in word_tags:
    no_of_tags = len(word_tags[word])
    if no_of_tags in tag_words_dict.keys():
        tag_words_dict[no_of_tags].append(word)
    else:
        temp = list()
        temp.append(word)
        tag_words_dict[no_of_tags] = temp

# Dictionary whose keys are no_of_tags where values are words as per the questions
final_dict = dict()
for no_of_tags in tag_words_dict.keys():
    final_dict[no_of_tags]=list()

keys = list(tag_words_dict.keys())
for no_of_tags in keys:
    for word in tag_words_dict[no_of_tags]:
        entry = list()
        # Add the Word
        entry.append(word)
        # Add its count
        entry.append(untagged_words_fdist[word])
        # Add its possible tags
        entry.append(word_tags[word])
        # Add its Entropy
        entry.append(tag_entropies_of_words[word])
        # Add the 4-tuple entry to final dictionary
        final_dict[no_of_tags].append(entry)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
for key, value in final_dict.items():
    print(key,len(value))



