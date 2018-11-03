import numpy as np
from numpy import array 
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, CuDNNLSTM, concatenate
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras import optimizers, initializers, constraints, regularizers
from keras.engine.topology import Layer
from tensorflow.keras.layers import Activation
from tensorflow.layers import Flatten
cb6133filename = 'cb6133filtered.npy'
cb513 = 'cb513.npy'
def load_augmented_data(npy_path, max_len):
    data = np.load(npy_path)
    residue_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
    q8_list = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq']
    data_reshape = data.reshape(data.shape[0], 700, -1)
    residue_onehot = data_reshape[:,:,0:22]
    residue_q8_onehot = data_reshape[:,:,22:31]
    profile = data_reshape[:,:,35:57]
    zero_arr = np.zeros((profile.shape[0], max_len - profile.shape[1], profile.shape[2]))
    profile_padded = np.concatenate([profile, zero_arr], axis=1)
    residue_array = np.array(residue_list)[residue_onehot.argmax(2)]
    q8_array = np.array(q8_list)[residue_q8_onehot.argmax(2)]
    residue_str_list = []
    q8_str_list = []
    for vec in residue_array:
        x = ''.join(vec[vec != 'NoSeq'])
        residue_str_list.append(x)
    for vec in q8_array:
        x = ''.join(vec[vec != 'NoSeq'])
        q8_str_list.append(x)
    id_list = np.arange(1, len(residue_array) + 1)
    len_list = np.array([len(x) for x in residue_str_list])
    train_df = pd.DataFrame({'id': id_list, 'len': len_list, 'input': residue_str_list, 'expected': q8_str_list})
    return train_df, profile_padded

def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

counter = 0
# prints the results
def print_results(x, y_, revsere_decoder_index, counter,test_df):
    Ans['id'][counter] = test_df['id'][counter]
    # print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    Ans['expected'][counter] = str(onehot_to_seq(y_, revsere_decoder_index).upper())

#
def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])


maxlen_seq = 700
train_df, X_aug_train = load_augmented_data(cb6133filename,maxlen_seq)
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
test_df, X_aug_test = load_augmented_data(cb513,maxlen_seq)
test_input_seqs, test_target_seqs = test_df[['input','expected']][(test_df.len <= maxlen_seq)].values.T






train_input_grams = seq2ngrams(train_input_seqs)
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
y_train = to_categorical(train_target_data)


test_input_grams = seq2ngrams(test_input_seqs)
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')
y_test = to_categorical(test_target_data)


n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

input = Input(shape = (maxlen_seq,))
input2 = Input(shape=(maxlen_seq,22))

x = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)
x = concatenate([x,input2],axis=2)
z = Conv1D(64, 11, strides=1, padding='same')(x)
w = Conv1D(64, 7, strides=1, padding='same')(x)
x = concatenate([x,z],axis=2)
x = concatenate([x,w],axis=2)
z = Conv1D(64, 5, strides=1, padding='same')(x)
w = Conv1D(64, 3, strides=1, padding='same')(x)
x = concatenate([x,z],axis=2)
x = concatenate([x,w],axis=2)
x = Bidirectional(CuDNNLSTM(units = 128, return_sequences = True))(x)

y = TimeDistributed(Dense(n_tags, activation = "softmax"))(x)

model = Model([input,input2], y)
model.summary()
model.compile(optimizer = 'RMSprop', loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])
model.fit([X_train,X_aug_train], y_train, batch_size = 128, epochs = 30, validation_data = ([X_test,X_aug_test], y_test), verbose = 1)
revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}
acc = model.evaluate([X_test,X_aug_test], y_test)
print (acc)
y_test_pred = model.predict([X_test,X_aug_test])
np.save('cb513_test_prob_6.npy', y_test_pred)
counter = 0
Ans = pd.DataFrame(0,index = np.arange(len(X_test)), columns = ['id','expected'])
for i in range(len(X_test)):
    print_results(X_test[i], y_test_pred[i], revsere_decoder_index,counter,test_df)
    counter+=1
Ans.to_csv('cb513_test_6.csv',index = False)