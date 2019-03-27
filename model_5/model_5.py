############################################
#
# LSTMs with Luang attention
#
############################################

##### Load .npy data file and generate sequence csv and profile csv files  #####
import numpy as np
import pandas as pd

def load_augmented_data(npy_path, max_len):
    data = np.load(npy_path)
    residue_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
    q8_list = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq']

    data_reshape = data.reshape(data.shape[0], 700, -1)
    residue_onehot = data_reshape[:,:,0:22]
    residue_q8_onehot = data_reshape[:,:,22:31]
    nc_terminal = data_reshape[:,:,31:33]
    profile = data_reshape[:,:,35:57]
    zero_arr = np.zeros((profile.shape[0],max_len - profile.shape[1], profile.shape[2]))
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

    profile_padded_wrapped = profile_padded.reshape((data.shape[0], 700*22))
    train_df = pd.DataFrame({'id': id_list, 'len': len_list, 'input': residue_str_list, 'expected': q8_str_list})
    return train_df, profile_padded_wrapped

cb513filename = '../data/cb513.npy'
cb6133filename = '../data/cb6133.npy'
cb6133filteredfilename = '../data/cb6133filtered.npy'

max_len =700
train_df, profile_padded_wrapped = load_augmented_data(cb6133filename, max_len)
train_df[['id', 'len','input','expected']].to_csv('cb6133.csv', sep=',', encoding='utf-8', index=False)
profile_df = pd.DataFrame(profile_padded_wrapped)
profile_df.to_csv('cb6133_profile.csv',index=False)

train_df, profile_padded_wrapped = load_augmented_data(cb513filename, max_len)
train_df[['id', 'len','input','expected']].to_csv('cb513.csv', sep=',', encoding='utf-8', index=False)
profile_df = pd.DataFrame(profile_padded_wrapped)
profile_df.to_csv('cb513_profile.csv',index=False)

train_df, profile_padded_wrapped = load_augmented_data(cb6133filteredfilename, max_len)
train_df[['id', 'len','input','expected']].to_csv('cb6133filtered.csv', sep=',', encoding='utf-8', index=False)
profile_df = pd.DataFrame(profile_padded_wrapped)
profile_df.to_csv('cb6133filtered_profile.csv',index=False)

################################################################################
# Run this part if you are running on your own machine and
# all data files are in the same dir with this file.
################################################################################

cb6133_df = pd.read_csv('cb6133.csv', sep=',')
cb6133_profile_df = pd.read_csv('cb6133_profile.csv', sep=',')
cb6133filtered_df = pd.read_csv('cb6133filtered.csv', sep=',')
cb6133filtered_profile_df = pd.read_csv('cb6133filtered_profile.csv', sep=',')
cb513_df = pd.read_csv('cb513.csv', sep=',')
cb513_profile_df = pd.read_csv('cb513_profile.csv', sep=',')

# cb6133test
# train_df, val_df, test_df = cb6133_df[0:5600], cb6133_df[5877:6133], cb6133_df[5605:5877]
# train_profile_df, val_profile_df, test_profile_df = cb6133_profile_df[0:5600], cb6133_profile_df[5877:6133], cb6133_profile_df[5605:5877]

# cb513test
train_df, val_df, test_df = cb6133filtered_df, cb6133_df[5877:6133], cb513_df
train_profile_df, val_profile_df, test_profile_df = cb6133filtered_profile_df, cb6133_profile_df[5877:6133], cb513_profile_df
################################################################################

################################################################################
# Uncomment and run this part instead if you are using colab and
# all the data files (sequence/profile csv) are uploaded to drive.
# Remember to replace the file IDs with the ones you uploaded.
# Copy & paste this cell to colab notebook and run.
################################################################################

# !pip install -U -q PyDrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# from io import StringIO

# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

# ########## A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz ##########

# # cb6133_file_id = '14z70nLvJThZ38j1sELLVwFIUy7aoXHGY'
# # cb6133_downloaded = drive.CreateFile({'id': cb6133_file_id})
# # cb6133_profile_file_id = '1EivJeG-NertakPMBWQcyLpawR-eR9r6d'
# # cb6133_profile_downloaded = drive.CreateFile({'id': cb6133_profile_file_id})
# cb6133filtered_file_id = '1u50UsRFbDZTzQzZ8BkVy8fcUPUN8SOJ7'
# cb6133filtered_downloaded = drive.CreateFile({'id': cb6133filtered_file_id})
# cb6133filtered_profile_file_id = '1mNuGfrEGaKhNmGNDt0MRW5bfYHGo4FI0'
# cb6133filtered_profile_downloaded = drive.CreateFile({'id': cb6133filtered_profile_file_id})
# cb513_file_id = '1t6GU8KwAiKt-J8mntTNZZJ8ZssvmjDYQ'
# cb513_downloaded = drive.CreateFile({'id': cb513_file_id})
# cb513_profile_file_id = '1J6f_WnMjTEKXebmdnadB4XDiP5873Evj'
# cb513_profile_downloaded = drive.CreateFile({'id': cb513_profile_file_id})

# # cb6133_data_str = StringIO(cb6133_downloaded.GetContentString())
# # cb6133_profile_data_str = StringIO(cb6133_profile_downloaded.GetContentString())
# cb6133filtered_data_str = StringIO(cb6133filtered_downloaded.GetContentString())
# cb6133filtered_profile_data_str = StringIO(cb6133filtered_profile_downloaded.GetContentString())
# cb513_data_str = StringIO(cb513_downloaded.GetContentString())
# cb513_profile_data_str = StringIO(cb513_profile_downloaded.GetContentString())

# import pandas as pd
# # cb6133_df = pd.read_csv(cb6133_data_str, sep=',')
# # cb6133_profile_df = pd.read_csv(cb6133_profile_data_str, sep=',')
# cb6133filtered_df = pd.read_csv(cb6133filtered_data_str, sep=',')
# cb6133filtered_profile_df = pd.read_csv(cb6133filtered_profile_data_str, sep=',')
# cb513_df = pd.read_csv(cb513_data_str, sep=',')
# cb513_profile_df = pd.read_csv(cb513_profile_data_str, sep=',')

# # cb6133test and cb6133 10fold cv
# # train_df, val_df, test_df = cb6133_df[0:5600], cb6133_df[5877:6133], cb6133_df[5605:5877]
# # train_profile_df, val_profile_df, test_profile_df = cb6133_profile_df[0:5600], cb6133_profile_df[5877:6133], cb6133_profile_df[5605:5877]

# # cb513test
# train_df, val_df, test_df = cb6133filtered_df, cb513_df, cb513_df
# train_profile_df, val_profile_df, test_profile_df = cb6133filtered_profile_df, cb513_profile_df, cb513_profile_df
################################################################################

import numpy as np
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.layers import Activation, BatchNormalization, dot, concatenate
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
import keras


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

# prints the results
def print_results(x, y_, revsere_decoder_index):
    # print("input     : " + str(x))
    # print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    print(str(onehot_to_seq(y_, revsere_decoder_index).upper()))

# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n = 3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])


maxlen_seq = 700

# Loading and converting the inputs to trigrams
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
val_input_seqs, val_target_seqs = val_df[['input', 'expected']][(val_df.len <= maxlen_seq)].values.T

train_input_grams = seq2ngrams(train_input_seqs, n=2)
val_input_grams = seq2ngrams(val_input_seqs, n=2)

# Same for test
test_input_seqs = test_df['input'].values.T
test_input_grams = seq2ngrams(test_input_seqs, n=2)

train_profile = np.array(train_profile_df).reshape((train_input_seqs.shape[0], 700, 22)) # shape = (5600, 700, 22)
val_profile = np.array(val_profile_df).reshape((val_input_seqs.shape[0], 700, 22))       # shape = (256, 700, 22)
test_profile = np.array(test_profile_df).reshape((test_input_seqs.shape[0], 700, 22))    # shape = (272, 700, 22)

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

# use same tokenizer defined on train for tokenization of validation
val_input_data = tokenizer_encoder.texts_to_sequences(val_input_grams)
val_input_data = sequence.pad_sequences(val_input_data, maxlen = maxlen_seq, padding='post')
val_target_data = tokenizer_decoder.texts_to_sequences(val_target_seqs)
val_target_data = sequence.pad_sequences(val_target_data, maxlen = maxlen_seq, padding='post')
val_target_data = to_categorical(val_target_data)

# Use the same tokenizer defined on train for tokenization of test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_input_data = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

############################### Model starts here ##############################

input = Input(shape = (maxlen_seq,))
embed_out = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)
profile_input = Input(shape = (maxlen_seq,22))
x = concatenate([embed_out, profile_input]) # 5600, 700, 150

x1_out = Bidirectional(LSTM(units = 75, return_sequences=True, recurrent_dropout=0.2), merge_mode='concat')(x)
x1_out_last = x1_out[:,-1,:]

x2_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x1_out, initial_state=[x1_out_last, x1_out_last])
x2_out_last = x2_out[:,-1,:]

attention = dot([x2_out, x1_out], axes=[2, 2])
attention = Activation('softmax')(attention)
context = dot([attention, x1_out], axes=[2, 1])
x2_out_combined_context = concatenate([context, x2_out])

x3_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x2_out, initial_state=[x2_out_last, x2_out_last])
x3_out_last = x3_out[:,-1,:]

attention_2 = dot([x3_out, x2_out], axes=[2, 2])
attention_2 = Activation('softmax')(attention_2)
context_2 = dot([attention_2, x2_out], axes=[2, 1])
x3_out_combined_context = concatenate([context_2, x3_out])

attention_2_1 = dot([x3_out, x1_out], axes=[2, 2])
attention_2_1 = Activation('softmax')(attention_2_1)
context_2_1 = dot([attention_2_1, x1_out], axes=[2, 1])
x3_1_out_combined_context = concatenate([context_2_1, x3_out])

x4_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x3_out, initial_state=[x3_out_last, x3_out_last])
x4_out_last = x4_out[:,-1,:]

attention_3 = dot([x4_out, x3_out], axes=[2, 2])
attention_3 = Activation('softmax')(attention_3)
context_3 = dot([attention_3, x3_out], axes=[2, 1])
x4_out_combined_context = concatenate([context_3, x4_out])

attention_3_1 = dot([x4_out, x2_out], axes=[2, 2])
attention_3_1 = Activation('softmax')(attention_3_1)
context_3_1 = dot([attention_3_1, x2_out], axes=[2, 1])
x4_1_out_combined_context = concatenate([context_3_1, x4_out])

attention_3_2 = dot([x4_out, x1_out], axes=[2, 2])
attention_3_2 = Activation('softmax')(attention_3_2)
context_3_2 = dot([attention_3_2, x1_out], axes=[2, 1])
x4_2_out_combined_context = concatenate([context_3_2, x4_out])

x5_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x4_out, initial_state=[x4_out_last, x4_out_last])
x5_out_last = x5_out[:,-1,:]

attention_4 = dot([x5_out, x4_out], axes=[2, 2])
attention_4 = Activation('softmax')(attention_4)
context_4 = dot([attention_4, x4_out], axes=[2, 1])
x5_out_combined_context = concatenate([context_4, x5_out])

attention_4_1 = dot([x5_out, x3_out], axes=[2, 2])
attention_4_1 = Activation('softmax')(attention_4_1)
context_4_1 = dot([attention_4_1, x3_out], axes=[2, 1])
x5_1_out_combined_context = concatenate([context_4_1, x5_out])

attention_4_2 = dot([x5_out, x2_out], axes=[2, 2])
attention_4_2 = Activation('softmax')(attention_4_2)
context_4_2 = dot([attention_4_2, x2_out], axes=[2, 1])
x5_2_out_combined_context = concatenate([context_4_2, x5_out])

attention_4_3 = dot([x5_out, x1_out], axes=[2, 2])
attention_4_3 = Activation('softmax')(attention_4_3)
context_4_3 = dot([attention_4_3, x1_out], axes=[2, 1])
x5_3_out_combined_context = concatenate([context_4_3, x5_out])

out = keras.layers.Add()([x2_out_combined_context, \
             x3_out_combined_context, x3_1_out_combined_context,\
             x4_out_combined_context, x4_1_out_combined_context, x4_2_out_combined_context, \
             x5_out_combined_context, x5_1_out_combined_context, x5_2_out_combined_context, x5_3_out_combined_context])

fc1_out = TimeDistributed(Dense(150, activation="relu"))(out) # equation (5) of the paper
output = TimeDistributed(Dense(n_tags, activation="softmax"))(fc1_out) # equation (6) of the paper

model = Model([input, profile_input], output)
model.summary()

################################################################################

# Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
rmsprop = keras.optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=None, decay=0.0) # add decay=0.5 after 15 epochs
model.compile(optimizer = rmsprop, loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])

# Splitting the data for train and validation sets
X_train, X_val, y_train, y_val = train_input_data, val_input_data, train_target_data, val_target_data


# Training the model on the training data and validating using the validation set
model.fit([X_train, train_profile], y_train, batch_size = 64, epochs = 20, validation_data = ([X_val, val_profile], y_val), verbose = 1)

# Defining the decoders so that we can
revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}

y_test_pred = model.predict([test_input_data, test_profile])
# print(len(test_input_data))
for i in range(len(test_input_data)):
    print_results(test_input_seqs[i], y_test_pred[i], revsere_decoder_index)


##### if running on local dir
prob_file = np.save('cb513_test_prob_5.npy', y_test_pred)

##### if using colab
# from google.colab import files
# file_ = np.save('cb513.npy', y_test_pred)
# files.download('cb513.npy')
