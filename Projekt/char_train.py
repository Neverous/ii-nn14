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
n_epochs = 1
learning_rate = 0.1
learning_rate_decay = 0.999
train_ratio = 2.0 / 3
hidden_ratio = 16.0 / 3
variance = 50

## CODE
print ':: CHAR-BASED TRAINING'
print ':::: READING DATASET'
from dataset_char_stats import *

print ':::: PREPARING MATRICES'
length = char_based_length_choice
bucket = list(map(lambda sentence: sentence[:char_based_length_choice], [item for _, seq in filter(lambda (length, sentence): length >= char_based_length_choice, buckets.items()) for item in seq]))
random.shuffle(bucket)

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

print ':::: STARTING TRAINING'
class CharBased(rnn_minibatch.MetaRNN): pass

model = CharBased(n_in=char_based_letters_count,
                  n_out=char_based_letters_count,
                  n_hidden=int(char_based_letters_count * hidden_ratio),
                  learning_rate=learning_rate,
                  learning_rate_decay=learning_rate_decay,
                  n_epochs=n_epochs,
                  activation='tanh',
                  output_type='softmax',
                  snapshot_every=10,
                  snapshot_path='snapshots/')

try:
    latest_snapshot = max(glob.iglob('snapshots/CharBased*.pkl'), key=os.path.getmtime)
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
model.save('models/char_model-final-sgd.pkl')
