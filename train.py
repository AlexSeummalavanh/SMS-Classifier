from models.rnn import message_rnn_model
from models.attention import message_attention_model
import numpy as np

if __name__ == "__main__":
    data_filepath = "data/sms_data.tsv"
    total_rows = np.arange(0, 5000)
    np.random.shuffle(total_rows)
    
    split = int(0.8 * len(total_rows))
    training_rows = total_rows[:split]
    validation_rows = total_rows[split:]

    print("Training RNN model...")
    rnn_model, rnn_train_perf, rnn_val_perf = message_rnn_model(data_filepath, training_rows, validation_rows)

    print("\nTraining Attention model...")
    attn_model, attn_train_perf, attn_val_perf = message_attention_model(data_filepath, training_rows, validation_rows)
