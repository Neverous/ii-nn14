import sys
import os
import glob
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)-8.8s - %(name)-8.8s - %(message)s')
logger = logging.getLogger()

import numpy
import theano
import rnn_minibatch
import pickle
import collections

## CONFIGURATION
n_epochs = 10000
learning_rate = 0.1
learning_rate_decay = 0.99
train_ratio = 99. / 100.
hidden_ratio = 16.0 / 3
variance = 50

## CODE
print ':: WORD-BASED TRAINING'
print ':::: READING DATASET'
from dataset_word_stats import *
random.shuffle(word_based_sentences)

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

print ':::: STARTING TRAINING'
class WordBased(rnn_minibatch.MetaRNN): pass

model = WordBased(n_in=word_based_words_count,
                  n_out=word_based_words_count,
                  n_hidden=int(word_based_words_count * hidden_ratio)
                  learning_rate=learning_rate,
                  learning_rate_decay=learning_rate_decay,
                  n_epochs=n_epochs,
                  activation='tanh',
                  output_type='softmax',
                  snapshot_every=10,
                  snapshot_path='snapshots/')

try:
    latest_snapshot = max(glob.iglob('snapshots/WordBased*.pkl'), key=os.path.getmtime)
except:
    latest_snapshot = None

try:
    print ':::::: SIZE: %dx%d' % (train_target.shape[0], train_target.shape[1])
    if latest_snapshot:
        print ':::::: LOADING SNAPSHOT'
        model.load(latest_snapshot)

    model.fit(train_data, train_target, verify_data, verify_target, validate_every=1, show_norms=False, show_output=False, optimizer='sgd')

except KeyboardInterrupt:
    pass

print ':::: SAVING MODEL'
model.save('models/word_model-final-sgd.pkl')
