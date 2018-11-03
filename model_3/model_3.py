
import sys
sys.path.append('keras-tcn')
from tcn import tcn
import h5py
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np 
import dill as pickle
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute
import tensorflow as tf
from keras.layers.merge import concatenate
#from google.colab import files
from keras.layers import Dropout
from keras import regularizers
from keras.layers import merge
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from keras.engine.topology import Layer


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import os
#uploaded = files.upload()
train_df = pd.read_csv('cb6133filtered.npy.csv')
test_df = pd.read_csv('cb513.npy.csv')

data_train = np.load('../data/cb6133filtered.npy')
data_reshape_train = data_train.reshape(data_train.shape[0], 700, -1)
profile_train = data_reshape_train[:,:,35:57]
zero_arr_train = np.zeros((profile_train.shape[0], 800 - profile_train.shape[1], profile_train.shape[2]))
profile_padded_train = np.concatenate([profile_train, zero_arr_train], axis=1)

data_test = np.load('../data/cb513.npy')
data_reshape_test = data_test.reshape(data_test.shape[0], 700, -1)
profile_test = data_reshape_test[:,:,35:57]
zero_arr_test = np.zeros((profile_test.shape[0], 800 - profile_test.shape[1], profile_test.shape[2]))
profile_padded_test = np.concatenate([profile_test, zero_arr_test], axis=1)


# The custom accuracy metric used for this task
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())
  
# Decode: map to a sequence from a one-hot 
# encoding, takes a one-hot encoded y matrix 
# with an lookup table "index"
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n = 2):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])
  


maxlen_seq = 800


# Loading and converting the inputs to trigrams
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
train_input_grams = seq2ngrams(train_input_seqs)

# Same for test
test_input_seqs = test_df['input'].values.T
test_input_grams = seq2ngrams(test_input_seqs)

# Initializing and defining the tokenizer encoders and decoders based on the train set
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

# Using the tokenizer to encode and decode the sequences for use in training
# Inputs
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
train_input_data = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')

# Targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
train_target_data = to_categorical(train_target_data)

# Use the same tokenizer defined on train for tokenization of test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_input_data = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1


def build_model():
  input = Input(shape = (None, ))
  profiles_input = Input(shape = (None, 22))

  # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
  x1 = Embedding(input_dim = n_words, output_dim = 250, input_length = None)(input)  
  x1 = concatenate([x1, profiles_input], axis = 2)
  
  x2 = Embedding(input_dim = n_words, output_dim = 125, input_length = None)(input)
  x2 = concatenate([x2, profiles_input], axis = 2)

  x1 = Dense(1200, activation = "relu")(x1)
  x1 = Dropout(0.5)(x1)

  # Defining a bidirectional LSTM using the embedded representation of the inputs
  x2 = Bidirectional(CuDNNGRU(units = 500, return_sequences = True))(x2)
  x2 = Bidirectional(CuDNNGRU(units = 100, return_sequences = True))(x2)
  COMBO_MOVE = concatenate([x1, x2])
  w = Dense(500, activation = "relu")(COMBO_MOVE) # try 500
  w = Dropout(0.4)(w)
  w = tcn.TCN()(w)
  y = TimeDistributed(Dense(n_tags, activation = "softmax"))(w)

  # Defining the model as a whole and printing the summary
  model = Model([input, profiles_input], y)
  #model.summary()

  # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
  adamOptimizer = Adam(lr=0.0025, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False) 
  model.compile(optimizer = adamOptimizer, loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])
  return model


# Defining the decoders so that we can
revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}


# prints the results
def print_results(id, y_, revsere_decoder_index):
    print('%s,%s' % (id, onehot_to_seq(y_, revsere_decoder_index).upper()))


def save_model(model):
  # serialize model to JSON
#   model_json = model.to_json()
#   with open("model.json", "w") as json_file:
#       json_file.write(model_json)
#   # serialize weights to HDF5
#   model.save_weights("model.h5")
  model.save('cb6133filtered.model.h5')
  print("Saved model to disk")

  #drive = GoogleDrive(gauth)
  #file1 = drive.CreateFile({'title': 'cb6133filtered.model.h5'})
  #file1.SetContentFile('cb6133filtered.model.h5')
  #file1.Upload() # Upload the file.


VERBOSE = 1
SAVE_MODEL = True
model = build_model()
model.fit([train_input_data, profile_padded_train], train_target_data, batch_size = 30, epochs = 6, verbose = VERBOSE, shuffle=True)


if SAVE_MODEL:
  save_model(model)

y_test_pred = model.predict([test_input_data[:], profile_padded_test])
ids = test_df['id'].values

print ('id,expected')
for i in range(len(test_df)):
  print_results(ids[i], y_test_pred[i], revsere_decoder_index)


# Loading and converting the inputs to trigrams
test_input_seqs, test_target_seqs = test_df[['input', 'expected']].values.T
tokenizer_decoder.fit_on_texts(test_target_seqs)

# Targets
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = to_categorical(test_target_data)


acc = accuracy(test_target_data, y_test_pred)
print ('accuracy on cb513:', tf.Session().run(acc).mean())

