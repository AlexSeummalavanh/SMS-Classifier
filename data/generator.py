import tensorflow as tf
import numpy as np
import pandas as pd 
import scipy as sp
from sklearn.model_selection import train_test_split
import math
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50
VOCAB_SIZE = 10000
# Keras data generator for the SMS Spam Collection dataset
class MessageDataGenerator(tf.keras.utils.Sequence):

  def __init__(self, filepath, rows, batch_size):
    # filepath is a full file path to the file containing the data
    # rows is the rows from the file to use
    # batch_size is the batch size
    self.batch_size = batch_size
    self.rows = rows
    # Read the data
    self.data = pd.read_csv(filepath, sep="\t", header=None, names=["label", "message"])
    # One-hot encoding
    self.data["label"] = self.data["label"].map({"ham": [1,0], "spam":[0,1]})
    # Tokenize
    self.tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    self.tokenizer.fit_on_texts(self.data["message"])  # Fit on selected rows
    self.indexes = np.arange(len(self.data))  # For shuffling

    # Return nothing    

  def __len__(self):
    #batches_per_epoch is the total number of batches used for one epoch
    batches_per_epoch = math.ceil(len(self.rows)/self.batch_size)
    return batches_per_epoch


  def __getitem__(self, index):
    # index is the index of the batch to be retrieved
    batch_indexes = self.rows[index * self.batch_size:(index + 1) * self.batch_size]
    batch_data = self.data.iloc[batch_indexes]

    # Tokenize and pad
    x = self.tokenizer.texts_to_sequences(batch_data["message"])
    x = pad_sequences(x, maxlen=MAX_LEN, padding='post', truncating='post')

    y = np.array(batch_data["label"].tolist())  # One-hot encoded labels

    # x is one batch of data
    # y is the labels for the batch of data
    return x, y