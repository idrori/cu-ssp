import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import os, pickle

from utils import *

maxlen_seq = 768
center_augmented_data = False ## center augmented data

training_idx = np.arange(5600)
test_idx = np.arange(5605,5877)
validation_idx = np.arange(5877,6133)

cb6133filename = '../data/cb6133.npy'


train_df_all, train_augment_data_all = load_augmented_data(cb6133filename, maxlen_seq, centered=center_augmented_data)
train_df = train_df_all.iloc[training_idx]
val_df = train_df_all.iloc[validation_idx]

# save preprocessed val and test data
test_df = train_df_all.iloc[test_idx]
test_augment_data = train_augment_data_all[test_idx]


# Loading and converting the inputs to ngrams
train_input_seqs, train_target_seqs = train_df_all[['input', 'expected']].values.T
train_input_grams = seq2ngrams(train_input_seqs, n=1)

# Initializing and defining the tokenizer encoders and decoders based on the train set
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

# Using the tokenizer to encode and decode the sequences for use in training
# Inputs
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
train_input_data = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post', truncating='post')

# Targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post', truncating='post')
train_target_data = to_categorical(train_target_data)

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

############################################
# Splitting the data for train and validation sets

X_val = train_input_data[validation_idx]
X_train = train_input_data[training_idx]
y_val = train_target_data[validation_idx]
y_train = train_target_data[training_idx]

X_train_augment = train_augment_data_all[training_idx]
X_val_augment = train_augment_data_all[validation_idx]

############################################
# save preprocessed val and test data and tokenizer

script_name = os.path.basename(__file__).split(".")[0] 
model_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + script_name
log_dir = '../logs/{}'.format(model_name)
os.mkdir(log_dir)

val_df.to_csv(os.path.join(log_dir, 'val_data.csv'))
np.save(os.path.join(log_dir, 'val_augment_data.npy'), X_val_augment)
test_df.to_csv(os.path.join(log_dir, 'test_data.csv'))
np.save(os.path.join(log_dir, 'test_augment_data.npy'), test_augment_data)

with open(os.path.join(log_dir, 'tokenizer_encoder.pickle'), 'wb') as handle:
    pickle.dump(tokenizer_encoder, handle)

with open(os.path.join(log_dir, 'tokenizer_decoder.pickle'), 'wb') as handle:
    pickle.dump(tokenizer_decoder, handle)

############################################
# Dropout to prevent overfitting. 
droprate = 0.25


def conv_block(x, n_channels, droprate):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x) 
    x = Dropout(droprate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x 

def up_block(x, n_channels):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling1D(size = 2)(x)
    x = Conv1D(n_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

input = Input(shape = (None, ))
augment_input = Input(shape = (None, 22))

# Defining an embedding layer mapping from the words (n_words) to a vector of len 128
embed_input = Embedding(input_dim = n_words, output_dim = 128, input_length = None)(input)

merged_input = concatenate([embed_input, augment_input], axis = 2)
merged_input = Conv1D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merged_input) 

conv1 = conv_block(merged_input, 128, droprate)
pool1 = MaxPooling1D(pool_size=2)(conv1)

conv2 = conv_block(pool1, 192, droprate)
pool2 = MaxPooling1D(pool_size=2)(conv2)

conv3 = conv_block(pool2, 384, droprate)
pool3 = MaxPooling1D(pool_size=2)(conv3)

conv4 = conv_block(pool3, 768, droprate)
pool4 = MaxPooling1D(pool_size=2)(conv4)

conv5 = conv_block(pool4, 1536, droprate)

up4 = up_block(conv5, 768)
up4 = concatenate([conv4,up4], axis = 2)
up4 = conv_block(up4, 768, droprate)

up3 = up_block(up4, 384)
up3 = concatenate([conv3,up3], axis = 2)
up3 = conv_block(up3, 384, droprate)

up2 = up_block(up3, 192)
up2 = concatenate([conv2,up2], axis = 2)
up2 = conv_block(up2, 192, droprate)

up1 = up_block(up2, 128)
up1 = concatenate([conv1,up1], axis = 2)
up1 = conv_block(up1, 128, droprate)

up1 = BatchNormalization()(up1)
up1 = ReLU()(up1)

# the following it equivalent to Conv1D with kernel size 1
# A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
y = TimeDistributed(Dense(n_tags, activation = "softmax"))(up1)


# Defining the model as a whole and printing the summary
model = Model([input, augment_input], y)
model.summary()

optim = RMSprop(lr=0.002)

def scheduler(i, lr):
    if i in [60]:
        return lr * 0.5
    return lr

reduce_lr = LearningRateScheduler(schedule=scheduler, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
#                             patience=8, min_lr=0.0005, verbose=1)

# Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
model.compile(optimizer = optim, loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])

tensorboard = TensorBoard(log_dir=log_dir)

checkpoint = ModelCheckpoint(os.path.join(log_dir, "best_val_acc.h5"),
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

# Training the model on the training data and validating using the validation set
model.fit([X_train, X_train_augment], y_train, batch_size = 128, 
            validation_data = ([X_val, X_val_augment], y_val), verbose = 1,
            callbacks=[tensorboard, reduce_lr, checkpoint], 
            epochs = 90)

K.clear_session()

model = load_model(os.path.join(log_dir, "best_val_acc.h5"))

val_pred_df = predict_all(model, val_df, tokenizer_encoder, tokenizer_decoder, n_gram=1,  max_len=maxlen_seq, 
                            augmented_input=X_val_augment,
                            filepath = os.path.join(log_dir, "val_pred_{}.csv".format(model_name)))
val_score, val_score_df = edit_score(val_df, val_pred_df,
                                    filepath = os.path.join(log_dir, "val_score_{}.csv".format(model_name)), plot=False)
plt.close()
K.clear_session()