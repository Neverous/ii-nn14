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

print ':::: PREPARING MATRICES'
length = char_based_length_choice
bucket = list(map(lambda sentence: sentence[:char_based_length_choice], [item for _, seq in filter(lambda (length, sentence): length >= char_based_length_choice, buckets.items()) for item in seq]))

train_count = int(numpy.ceil(len(bucket) * train_ratio))
verify_start = int(numpy.floor(len(bucket) * train_ratio))
# length x train_count x char_based_letters_count
train_data = numpy.zeros((length, train_count, char_based_letters_count), dtype=theano.config.floatX)
# length x train_count
train_target = numpy.zeros((length, train_count), dtype=numpy.int8)

# length x len(bucket) - verify_start + 1 x char_based_letters_count
verify_data = numpy.zeros((length, len(bucket) - verify_start + 1, char_based_letters_count), dtype=theano.config.floatX)
# length x len(bucket) - verify_start + 1
verify_target = numpy.zeros((length, len(bucket) - verify_start + 1), dtype=numpy.int8)

for s in xrange(train_count):
    for l in xrange(length):
        train_data[l, s, :] = numpy.random.uniform(size=char_based_letters_count)
        train_data[l, s, :] /= variance * (sum(train_data[l, s, :]) - train_data[l, s, char_based_letters[bucket[s][l]]])
        train_data[l, s, char_based_letters[bucket[s][l]]] = (variance - 1.0) / variance
        train_target[l, s] = char_based_letters[bucket[s][(l+1) % length]]

for s in xrange(verify_start, len(bucket)):
    for l in xrange(length):
        verify_data[l, s - verify_start, :] = numpy.random.uniform(size=char_based_letters_count)
        verify_data[l, s - verify_start, :] /= variance * (sum(verify_data[l, s - verify_start, :]) - verify_data[l, s - verify_start, char_based_letters[bucket[s][l]]])
        verify_data[l, s - verify_start, char_based_letters[bucket[s][l]]] = (variance - 1.0) / variance
        verify_target[l, s - verify_start] = char_based_letters[bucket[s][(l+1)%length]]

print ':::: TESTING'
class CharBased(rnn_minibatch.MetaRNN): pass
model = CharBased()
model.load(sys.argv[1])
print "Test error rate:", 100.0 * numpy.count_nonzero(model.predict(verify_data) - verify_target) / verify_target.size

for index in xrange(int(sys.argv[2])):
    print ''.join(map(char_based_letters_rev.__getitem__, model.predict(verify_data).T[index]))
    print ''.join(map(char_based_letters_rev.__getitem__, verify_target.T[index]))
    print ''
