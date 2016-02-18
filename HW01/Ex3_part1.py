"""
Author: Gaddipati, Asish Kumar
Ex3_part1.py solves a,b,c parts of Ex3
    1. Compute the % of occurrences of hapaxes and dis legomenas
    2. find the longest hapax and dis legomena
    3. plot r vs Nr
    4. plot log(r) vs log(Nr)

Run this with python3
Output:
Total No of Tokens in Brown Corpus  1161192
<FreqDist with 49815 samples and 1161192 outcomes>
Percentage of Hapax Legomena Occurrences 1.8954660383468023
Longest happax Legomena's are ['nnuolapertar-it-vuh-karti-birifw-']
Percentage of Dis Legomena Occurrences 1.2383826275069068
Longest Dis Legomena's are  ['definition-specialization']
"""

# Import all the necessary packages
from nltk.probability import FreqDist
from nltk.corpus import brown
import matplotlib.pyplot as plot
import pylab
from math import log

# Get the case insensitive words from the brown corpus
case_inses_words = [word.lower() for word in brown.words()]
no_of_tokens = len(case_inses_words)
print("Total No of Tokens in Brown Corpus ", no_of_tokens)

# Pass it on to FreqDist to get Frequency Distributions
fdist = FreqDist(case_inses_words)
print(fdist)

# Compute the Percentage of Hapax Legomena's Occurrences and the longest in them
hapax_legomenas = fdist.hapaxes() # Get the list of words that appeared just once in corpus
hapax_legomena_counts = len(hapax_legomenas) # Get the count of them
percentage_of_hapax_legomena = (hapax_legomena_counts/no_of_tokens)*100 # Compute percentage
print("Percentage of Hapax Legomena Occurrences", percentage_of_hapax_legomena)
max_len_happax_legomena = max([len(word) for word in hapax_legomenas])
print("Longest happax Legomena's are", [word for word in hapax_legomenas if len(word) == max_len_happax_legomena])

# Compute the Percentage of dis legomena Occurrences and the longest in them
dis_legomenas = [key for key, value in fdist.items() if value == 2] # Get the words that occurred just twice
dis_legomena_counts = len(dis_legomenas) * 2 # Get their counts
percentage_of_dis_legomena = (dis_legomena_counts/no_of_tokens)*100 # Compute percentage
print("Percentage of Dis Legomena Occurrences", percentage_of_dis_legomena)
max_len_dis_legomena = max([len(word) for word in dis_legomenas])
print("Longest Dis Legomena's are ", [word for word in dis_legomenas if len(word) == max_len_dis_legomena])

# Plot the r vs Nr graph
fdist.plot(50)

# Compute the log scaled version of r vs Nr
log_rvsNr = {log(key):log(value) for key, value in (fdist.r_Nr()).items() if value!=0}

# Plot the graph of log(r) vs log(Nr)
plot.plot(log_rvsNr.keys(), log_rvsNr.values(), 'r.')
plot.axis([-1, 11, -1, 11])
plot.xlabel('log(r)')
plot.ylabel('log(Nr)')
plot.title('log(r) vs log(Nr) Brown Corpus')
plot.show()

