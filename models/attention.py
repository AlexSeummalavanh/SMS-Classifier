from generators.message_data_generator import MessageDataGenerator
from utils.constants import MAX_LEN, VOCAB_SIZE
import tensorflow as tf

def message_attention_model(data_filepath, training_rows, validation_rows):
    train_gen = MessageDataGenerator(data_filepath, training_rows, 20)
    val_gen = MessageDataGenerator(data_filepath, validation_rows, 20)

    inputs = tf.keras.Input(shape=(MAX_LEN,))
    x = tf.keras.layers.Embedding(VOCAB_SIZE, 128)(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="A5_Attention_Model")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

    return model, model.evaluate(train_gen), model.evaluate(val_gen)