"""
Cascaded Convolution Model

- Pranav Shrestha (ps2958)
- Jeffrey Wan (jw3468)

"""

import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Embedding, Dense, TimeDistributed, Concatenate, BatchNormalization
from keras.layers import Bidirectional, Activation, Dropout, CuDNNGRU, Conv1D

from sklearn.model_selection import train_test_split, KFold
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.regularizers import l1, l2
import tensorflow as tf

### Data Retrieval
# cb6133         = np.load("../data/cb6133.npy")
cb6133filtered = np.load("../data/cb6133filtered.npy")
cb513          = np.load("../data/cb513.npy")

print()
# print(cb6133.shape)
print(cb6133filtered.shape)
print(cb513.shape)

maxlen_seq = r = 700 # protein residues padded to 700
f = 57  # number of features for each residue

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']

columns = ["id", "len", "input", "profiles", "expected"]

def get_data(arr, bounds=None):
    
    if bounds is None: bounds = range(len(arr))
    
    data = [None for i in bounds]
    for i in bounds:
        seq, q8, profiles = '', '', []
        for j in range(r):
            jf = j*f
            
            # Residue convert from one-hot to decoded
            residue_onehot = arr[i,jf+0:jf+22]
            residue = residue_list[np.argmax(residue_onehot)]

            # Q8 one-hot encoded to decoded structure symbol
            residue_q8_onehot = arr[i,jf+22:jf+31]
            residue_q8 = q8_list[np.argmax(residue_q8_onehot)]

            if residue == 'NoSeq': break      # terminating sequence symbol

            nc_terminals = arr[i,jf+31:jf+33] # nc_terminals = [0. 0.]
            sa = arr[i,jf+33:jf+35]           # sa = [0. 0.]
            profile = arr[i,jf+35:jf+57]      # profile features
            
            seq += residue # concat residues into amino acid sequence
            q8  += residue_q8 # concat secondary structure into secondary structure sequence
            profiles.append(profile)
        
        data[i] = [str(i+1), len(seq), seq, np.array(profiles), q8]
    
    return pd.DataFrame(data, columns=columns)

### Train-test Specification
train_df = get_data(cb6133filtered)
test_df  = get_data(cb513)

# The custom accuracy metric used for this task
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

def seq2onehot(seq, n):
    out = np.zeros((len(seq), maxlen_seq, n))
    for i in range(len(seq)):
        for j in range(maxlen_seq):
            out[i, j, seq[i, j]] = 1
    return out

# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

# Loading and converting the inputs to trigrams
train_input_seqs, train_target_seqs = \
    train_df[['input', 'expected']][(train_df.len.astype(int) <= maxlen_seq)].values.T
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
train_input_data = sequence.pad_sequences(train_input_data,
                                          maxlen = maxlen_seq, padding='post')

# Targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data,
                                           maxlen = maxlen_seq, padding='post')
train_target_data = to_categorical(train_target_data)

# Use the same tokenizer defined on train for tokenization of test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_input_data = sequence.pad_sequences(test_input_data,
                                         maxlen = maxlen_seq, padding='post')

# Computing the number of words and number of tags for the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

train_input_data_alt = train_input_data
train_input_data = seq2onehot(train_input_data, n_words)
train_profiles = train_df.profiles.values

test_input_data_alt = test_input_data
test_input_data = seq2onehot(test_input_data, n_words)
test_profiles = test_df.profiles.values

train_profiles_np = np.zeros((len(train_profiles), maxlen_seq, 22))
for i, profile in enumerate(train_profiles):
    for j in range(profile.shape[0]):
        for k in range(profile.shape[1]):
            train_profiles_np[i, j, k] = profile[j, k]

test_profiles_np = np.zeros((len(test_profiles), maxlen_seq, 22))
for i, profile in enumerate(test_profiles):
    for j in range(profile.shape[0]):
        for k in range(profile.shape[1]):
            test_profiles_np[i, j, k] = profile[j, k]

def decode_results(y_, reverse_decoder_index):
    print("prediction: " + str(onehot_to_seq(y_, reverse_decoder_index).upper()))
    return str(onehot_to_seq(y_, reverse_decoder_index).upper())

def run_test(_model, data1, data2, data3, csv_name, npy_name):
    reverse_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
    reverse_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}
    
    # Get predictions using our model
    y_test_pred = _model.predict([data1, data2, data3])

    decoded_y_pred = []
    for i in range(len(test_input_data)):
        res = decode_results(y_test_pred[i], reverse_decoder_index)
        decoded_y_pred.append(res)

    # Set Columns
    out_df = pd.DataFrame()
    out_df["id"] = test_df.id.values
    out_df["expected"] = decoded_y_pred

    # Save results
    with open(csv_name, "w") as f:
        out_df.to_csv(f, index=False)

    np.save(npy_name, y_test_pred)


""" Run below for a single run """
def train(X_train, y_train, X_val=None, y_val=None):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    model = CNN_BIGRU()
    model.compile(
        optimizer="Nadam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy", accuracy])
    
    if X_val is not None and y_val is not None:
        history = model.fit( X_train, y_train,
            batch_size = 128, epochs = 100,
            validation_data = (X_val, y_val))
    else:
        history = model.fit( X_train, y_train,
            batch_size = 128, epochs = 100)

    return history, model

""" Build model """
def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
    cnn = Conv1D(64, 11, padding="same")(x)
    if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
    if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
    if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
    if res:        cnn = Concatenate(axis=-1)([x, cnn])
    
    return cnn

def super_conv_block(x):
    c3 = Conv1D(32, 1, padding="same")(x)
    c3 = TimeDistributed(Activation("relu"))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)
    
    c7 = Conv1D(64, 3, padding="same")(x)
    c7 = TimeDistributed(Activation("relu"))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)
    
    c11 = Conv1D(128, 5, padding="same")(x)
    c11 = TimeDistributed(Activation("relu"))(c11)
    c11 = TimeDistributed(BatchNormalization())(c11)
    
    x = Concatenate(axis=-1)([x, c3, c7, c11])
    x = TimeDistributed(Dropout(0.5))(x)
    return x

def CNN_BIGRU():
    # Inp is one-hot encoded version of inp_alt
    inp          = Input(shape=(maxlen_seq, n_words))
    inp_alt      = Input(shape=(maxlen_seq,))
    inp_profiles = Input(shape=(maxlen_seq, 22))

    # Concatenate embedded and unembedded input
    x_emb = Embedding(input_dim=n_words, output_dim=64, 
                      input_length=maxlen_seq)(inp_alt)
    x = Concatenate(axis=-1)([inp, x_emb, inp_profiles])

    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)

    x = Bidirectional(CuDNNGRU(units = 256, return_sequences = True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation = "relu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    
    y = TimeDistributed(Dense(n_tags, activation = "softmax"))(x)
    
    model = Model([inp, inp_alt, inp_profiles], y)
    
    return model

X_train = [train_input_data, train_input_data_alt, train_profiles_np]
y_train = train_target_data

history, model = train(X_train, y_train)

# Save the model as a JSON format
model.save_weights("cb513_weights_1.h5")
with open("model_1.json", "w") as json_file:
    json_file.write(model.to_json())

# Save training history for parsing
with open("history_1.pkl", "wb") as hist_file:
    pickle.dump(history.history, hist_file)

# Predict on test dataset and save the output
run_test(model,
    test_input_data[:],
    test_input_data_alt[:],
    test_profiles_np[:],
    "cb513_test_1.csv", "cb513_test_prob_1.npy")
""" End single run """
