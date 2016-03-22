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
print ':: WORD-BASED TESTING'
print ':::: READING DATASET'
from dataset_word_stats import *

print ':::: GENERATOR'
class WordBased(rnn_minibatch.MetaRNN): pass
model = WordBased()
model.load(sys.argv[1])

gen_data = numpy.zeros((len(sys.argv[2:]), 1, word_based_words_count), dtype=theano.config.floatX)
for d, word in enumerate(sys.argv[2:]):
    gen_data[d, 0, word_based_words_rev[word]] = 1.0

for l in xrange(10):
    output = model.predict_proba(gen_data)
    gen_data = numpy.resize(gen_data, (l + 1 + len(sys.argv[2:]), 1, word_based_words_count))
    gen_data[l+len(sys.argv[2:])] = output[l-1+len(sys.argv[2:])]

print ' '.join(map(word_based_words.__getitem__, model.predict(gen_data).T[0]))
