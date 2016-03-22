import sys
import os
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)-8.8s - %(name)-8.8s - %(message)s')
import glob
import numpy
import theano
import rnn_minibatch
import pickle
import collections
import random

## CONFIGURATION
train_ratio = 2.0 / 3
variance = 50

## CODE
print ':: CHAR-BASED TESTING'
print ':::: READING DATASET'
from dataset_char_stats import *

print ':::: GENERATOR'
class CharBased(rnn_minibatch.MetaRNN): pass
model = CharBased()
model.load(sys.argv[1])

gen_data = numpy.zeros((len(sys.argv[2]), 1, char_based_letters_count), dtype=theano.config.floatX)
for d, letter in enumerate(sys.argv[2]):
    gen_data[d, 0, char_based_letters[letter]] = 1.0

for l in xrange(60):
    output = model.predict_proba(gen_data)
    gen_data = numpy.resize(gen_data, (l + 1 + len(sys.argv[2:]), 1, char_based_letters_count))
    gen_data[l+len(sys.argv[2:])] = output[l-1+len(sys.argv[2:])]

print ''.join(map(char_based_letters_rev.__getitem__, model.predict(gen_data).T[0]))
