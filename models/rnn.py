from data.generator import MessageDataGenerator
from utils.constants import MAX_LEN, VOCAB_SIZE
import tensorflow as tf

def message_rnn_model(data_filepath, training_rows, validation_rows):
    train_gen = MessageDataGenerator(data_filepath, training_rows, 20)
    val_gen = MessageDataGenerator(data_filepath, validation_rows, 20)

    model = tf.keras.Sequential(name="RNN")
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

    return model, model.evaluate(train_gen), model.evaluate(val_gen)