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
print ':: WORD-BASED TRAINING'
print ':::: READING DATASET'
from dataset_word_stats import *

print ':::: PREPARING MATRICES'
length = word_based_max_sentence_length

train_count = int(numpy.ceil(word_based_sentences_count * train_ratio))
verify_start = int(numpy.floor(word_based_sentences_count * train_ratio))
# length x train_count x word_based_words_count
train_data = numpy.zeros((length, train_count, word_based_words_count), dtype=theano.config.floatX)
# length x train_count
train_target = numpy.zeros((length, train_count), dtype=numpy.int16)

# length x word_based_sentences_count - verify_start + 1 x word_based_words_count
verify_data = numpy.zeros((length, word_based_sentences_count - verify_start + 1, word_based_words_count), dtype=theano.config.floatX)
# length x word_based_sentences_count - verify_start + 1
verify_target = numpy.zeros((length, word_based_sentences_count - verify_start + 1), dtype=numpy.int16)

for s in xrange(train_count):
    for l in xrange(length):
        train_data[l, s, :] = numpy.random.uniform(size=word_based_words_count)
        train_data[l, s, :] /= variance * (sum(train_data[l, s, :]) - train_data[l, s, word_based_sentences[s][l % len(word_based_sentences[s])]])
        train_data[l, s, word_based_sentences[s][l % len(word_based_sentences[s])]] = (variance - 1.0) / variance
        train_target[l, s] = word_based_sentences[s][(l+1) % len(word_based_sentences[s])]

for s in xrange(verify_start, word_based_sentences_count):
    for l in xrange(length):
        verify_data[l, s - verify_start, :] = numpy.random.uniform(size=word_based_words_count)
        verify_data[l, s - verify_start, :] /= variance * (sum(verify_data[l, s - verify_start, :]) - verify_data[l, s - verify_start, word_based_sentences[s][l % len(word_based_sentences[s])]])
        verify_data[l, s - verify_start, word_based_sentences[s][l % len(word_based_sentences[s])]] = (variance - 1.0) / variance
        verify_target[l, s - verify_start] = word_based_sentences[s][(l+1) % len(word_based_sentences[s])]

print ':::: TESTING'
class WordBased(rnn_minibatch.MetaRNN): pass
model = WordBased()
model.load(sys.argv[1])
print "Test error rate:", 100.0 * numpy.count_nonzero(model.predict(verify_data) - verify_target) / verify_target.size

for index in xrange(int(sys.argv[2])):
    print ' '.join(map(word_based_words.__getitem__, model.predict(verify_data).T[index]))
    print ' '.join(map(word_based_words.__getitem__, verify_target.T[index]))
    print ''
