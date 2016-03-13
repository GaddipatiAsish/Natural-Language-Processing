from collections import OrderedDict, Counter

"""
x = range(3)
y = ['a','b','c']
ziped = OrderedDict(zip(x, y))
for key, val in ziped.items():
    print(key,val)
"""
"""
import operator
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_x)
"""

x = (('a','1'),('a','1'),('b','2'))
print(Counter(x))