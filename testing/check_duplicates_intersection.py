import numpy as np
import pandas as pd
import sys

def duplicates(a):
	print(len(a), len(set(a)))
	return len(a) != len(set(a))

def intersection(a, b):
	return not set(a).isdisjoint(b)

trainfile = sys.argv[1]
testfile =  sys.argv[2]

train = pd.read_csv(trainfile)
trainsequences = list(train['sequence'])

test = pd.read_csv(testfile)
testsequences = list(test['sequence'])

print(duplicates(trainsequences))
print(duplicates(testsequences))
print(intersection(trainsequences, testsequences))