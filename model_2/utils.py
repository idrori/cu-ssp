import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing import sequence
import Levenshtein
import pickle

# The custom accuracy metric used for this task
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index, length=None):
    s = ''
    if length is None:
        for idx, o in enumerate(oh_seq):
            i = np.argmax(o)
            if i != 0:
                s += index[i]
            else:
                break
    else:
        for idx, o in enumerate(oh_seq):
            i = np.argmax(o[1:])
            if idx < length:
                s += index[i+1]
            else:
                break
    return s

# prints the results
def print_results(x, y_, revsere_decoder_index):
    # print("input     : " + str(x))
    # print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    print(str(onehot_to_seq(y_, revsere_decoder_index).upper()))

def decode_predictions(y_, revsere_decoder_index, length=None):
    return str(onehot_to_seq(y_, revsere_decoder_index, length=length).upper())


def predict_all(model, test_df, tokenizer_encoder, tokenizer_decoder, n_gram, augmented_input=None, max_len=None, filepath="submission.csv"):
    test_input_ids = test_df['id'].values
    test_input_seqs = test_df['input'].values.T
    test_input_grams = seq2ngrams(test_input_seqs, n=n_gram)
    
    revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}

    if max_len is None:
        max_test_length = max([len(x) for x in test_input_grams])
    else:
        max_test_length = max_len 
    test_input_data_full = tokenizer_encoder.texts_to_sequences(test_input_grams)
    test_input_data_full = sequence.pad_sequences(test_input_data_full, maxlen = max_test_length, padding = 'post')
    if augmented_input is None:
        y_test_pred = model.predict(test_input_data_full[:])
    else:
        y_test_pred = model.predict([test_input_data_full[:], augmented_input])
    np.save(filepath.replace(".csv", "_raw_pred.npy"), y_test_pred)
    y_test_pred_decoded = []
    for i in range(len(y_test_pred)):
        decoded = decode_predictions(y_test_pred[i], revsere_decoder_index, length=len(test_input_grams[i]))
        y_test_pred_decoded.append(decoded)
    test_pred_df = pd.DataFrame({'id':test_input_ids, "expected": y_test_pred_decoded},
                                columns = ['id', 'expected'])
    if np.all(np.array([len(x) for x in test_pred_df['expected']]) == np.array([len(x) for x in test_df['input']])):
        print("All length match")
    else:
        print("Some lengths do not match!")
    test_pred_df.to_csv(filepath, index=False)
    return test_pred_df

def ham_distance(x, y):
    return np.sum([a != b for a, b in zip(x, y)])

def edit_score(input_df, pred_df, filepath="edit_score.csv", plot=True):
    assert np.all(input_df['id'].values == pred_df['id'].values)
    if not np.all(np.array([len(x) for x in pred_df['expected']]) == np.array([len(x) for x in input_df['input']])):
        print("Some lengths do not match!")
        return None, None 
    output_df = input_df.copy().reset_index(drop=True)
    lev_dist = [Levenshtein.distance(x, y) for x, y in zip(input_df['expected'], pred_df['expected'])]
    ham_dist = [ham_distance(x, y) for x, y in zip(input_df['expected'], pred_df['expected'])]
    lev_score = np.mean(lev_dist)
    ham_score = np.mean(ham_dist)

    total_ham = np.sum(ham_dist)
    total_len = input_df['expected'].map(len).sum()
    accuracy = 1 - total_ham / total_len

    output_df['predicted'] = pred_df['expected'].values
    output_df['levdist'] = np.array(lev_dist)
    output_df['hamdist'] = np.array(ham_dist)
    output_df['levpercent'] = output_df['levdist'] / output_df['len']
    output_df['hampercent'] = output_df['hamdist'] / output_df['len']
    output_df['accuracy'] = 1 - output_df['hampercent']
    ham_percent = np.mean(output_df['hampercent'])
    mean_acc = np.mean(output_df['accuracy'])

    output_df.to_csv(filepath, index=False)
    print_str = "total acc: {:.4f}, mean acc: {:.4f}, lev: {:.1f}, ham: {:.1f}".format(accuracy, mean_acc, lev_score, ham_score)
    print(print_str)
    output_df.plot("len", "accuracy", kind="scatter")
    plt.hlines(y=accuracy, xmin=0, xmax=output_df['len'].max())
    plt.title(print_str)
    plt.savefig(filepath.replace(".csv", "_plot.png"))
    if plot:
        plt.show()
    plt.close()
    return accuracy, output_df

# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n = 3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

def load_augmented_data(npy_path, max_len, centered=False):
    data = np.load(npy_path)
    residue_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
    q8_list = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq']

    data_reshape = data.reshape(data.shape[0], 700, -1)
    residue_onehot = data_reshape[:,:,0:22]
    residue_q8_onehot = data_reshape[:,:,22:31]
    profile = data_reshape[:,:,35:57]
    # if centered:
    #     profile = profile - 0.5  # range [0,1]

    if max_len > profile.shape[1]:
        zero_arr = np.zeros((profile.shape[0], max_len - profile.shape[1], profile.shape[2]))
        zero_arr[:,:,-1] = 1.0
        profile_padded = np.concatenate([profile, zero_arr], axis=1)
    else:
        profile_padded = profile

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
