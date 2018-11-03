import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.metrics import categorical_accuracy
from keras import backend as K
from sklearn.model_selection import KFold
import tensorflow as tf

'''
various helper functions
'''

# Fixed-size Ordinally Forgetting Encoding
def encode_FOFE(onehot, alpha, maxlen):
    enc = np.zeros((maxlen, 2 * 22))
    enc[0, :22] = onehot[0]
    enc[maxlen-1, 22:] = onehot[maxlen-1]
    for i in range(1, maxlen):
        enc[i, :22] = enc[i-1, :22] * alpha + onehot[i]
        enc[maxlen-i-1, 22:] = enc[maxlen-i, 22:] * alpha + onehot[maxlen-i-1]
    return enc

# The custom accuracy metric used for this task
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

# Convert probabilities to secondary structure
def to_seq(y):
    seqs=[]
    for i in range(len(y)):
        seq_i=''
        for j in range(len(y[i])):
            seq_i += q8_list[np.argmax(y[i][j])]
        seqs.append(seq_i)
    return seqs

'''
Getting Data
'''
cb513filename = 'cb513.npy'
cb6133filename = 'cb6133.npy'
cb6133filteredfilename = 'cb6133filtered.npy'

cb513 = np.load(cb513filename)
cb6133 = np.load(cb6133filename)
cb6133filtered = np.load(cb6133filteredfilename)

residue_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', \
                'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
q8_list = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq']

r = 700 # protein residues padded to 700
f = 57 # number of features for each residue

'''
Setting up training, validation, test data
'''

maxlen_seq = 700 # maximum sequence length
alpha = 0.5 # parameter for long range encoding

train_input_seqs = cb6133filtered.reshape(5534, 700, 57)
train_input_data = np.zeros((5534, 700, 46))
train_input_data[:, :, :22] = train_input_seqs[:, :, :22]
train_input_data[:, :, 22:24] = train_input_seqs[:, :, 31:33]
train_input_data[:, :, 24:] = train_input_seqs[:, :, 35:]

train_target_data = train_input_seqs[:,:,22:31]

train_input_onehot = train_input_data[:,:,0:22]

train_input_fofe = np.array(list(map(lambda x:encode_FOFE(x, alpha, maxlen_seq),
                                     train_input_onehot)))
train_input_data = np.concatenate((train_input_data,
                                   train_input_fofe), axis=2)

test_input_seqs = cb513.reshape(514, 700, 57)
test_input_data = np.zeros((514, 700, 46))

test_input_data[:,:,:22] = test_input_seqs[:,:, :22]
test_input_data[:,:,22:24] = test_input_seqs[:,:, 31:33]
test_input_data[:,:,24:] = test_input_seqs[:,:, 35:]

test_input_onehot = test_input_data[:,:,0:22]
test_input_fofe = np.array(list(map(lambda x:encode_FOFE(x, alpha, maxlen_seq),
                                     test_input_onehot)))
test_input_data = np.concatenate((test_input_data,
                                   test_input_fofe), axis=2)

test_target_data = test_input_seqs[:,:,22:31]

# Computing the number of words and number of tags
n_words = len(train_input_data[0,0])
n_tags = len(train_target_data[0,0])
print(n_words,n_tags)
print(train_target_data.shape, train_input_data.shape, test_input_data.shape)

'''
Model
'''
input = Input(shape=(maxlen_seq, n_words,))

# one dense layer to remove sparsity
x = Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(input)
x = Reshape([maxlen_seq, 128, 1])(x)

# Defining 3 convolutional layers with different kernel sizes
# kernel size = 3
conv1 = ZeroPadding2D((3//2, 0), data_format='channels_last')(x)
conv1 = Conv2D(filters=64,
               kernel_size=(3, 128),
               input_shape=(1, maxlen_seq, 128),
               data_format='channels_last',
               strides=(1, 1),
               dilation_rate=(1, 1),
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv1)
conv1 = BatchNormalization(axis=-1)(conv1)

# kernel size = 7
conv2 = ZeroPadding2D((7//2, 0), data_format='channels_last')(x)
conv2 = Conv2D(filters=64,
               kernel_size=(7, 128),
               input_shape=(1, maxlen_seq, 128),
               data_format='channels_last',
               strides=(1, 1),
               padding='valid',
               dilation_rate=(1, 1),
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv2)
conv2 = BatchNormalization(axis=-1)(conv2)

# kernel size = 11
conv3 = ZeroPadding2D((11//2, 0), data_format='channels_last')(x)
conv3 = Conv2D(filters=64,
               kernel_size=(11, 128),
               input_shape=(1, maxlen_seq, 128),
               data_format='channels_last',
               strides=(1, 1),
               padding='valid',
               dilation_rate=(1, 1),
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv3)
conv3 = BatchNormalization(axis=-1)(conv3)
conv = concatenate([conv1, conv2, conv3])
conv = Reshape([maxlen_seq, 3*64])(conv)

# Defining 3 bidirectional GRU layers; taking the concatenation of outputs
gru1 = Bidirectional(GRU(32,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='hard_sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.0,
                         recurrent_dropout=0.1,
                         implementation=1))(conv)

gru2 = Bidirectional(GRU(32,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='hard_sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.0,
                         recurrent_dropout=0.1,
                         implementation=1))(gru1)

gru3 = Bidirectional(GRU(32,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='hard_sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.0,
                         recurrent_dropout=0.1,
                         implementation=1))(gru2)

comb = concatenate([gru1, gru2, gru3, conv])

# Defining two fully-connected layers with dropout
x = TimeDistributed(Dense(256,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(comb)
x = Dropout(0.1)(x)

x = TimeDistributed(Dense(128,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(x)
x = Dropout(0.1)(x)

# Defining the output layer
y = TimeDistributed(Dense(n_tags,
                          activation='softmax',
                          use_bias=False,
                          kernel_initializer='glorot_uniform'))(x)

# Defining the model as a whole and printing the summary
model = Model(input, y)
model.summary()

'''
Fitting and Predicting
'''
train_index = np.array(list(range(5534)))
test_index = np.array(list(range(514)))

X_train = train_input_data[train_index]
y_train = train_target_data[train_index]
X_test = test_input_data[test_index]

model.compile(optimizer = "nadam", loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])
model.fit(X_train, y_train, batch_size = 64, epochs = 12, verbose = 1)

y_pre = model.predict(X_test[:])
seq_pre = to_seq(y_pre)

path = 'cb513_test_5.csv'
file_output = pd.DataFrame({'id' : test_index+1, 'prediction' : seq_pre}, columns=['id', 'prediction'])
file_output.to_csv(path, index=False)
