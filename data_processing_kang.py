import numpy as np
import pandas as pd
import pickle

import torch as th
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def create_sequences(data, seq_length, stride, change_type='pct'):
    # Calculate changes
    if change_type == 'pct':
        # Percentage changes
        changes = pd.Series(data).pct_change(fill_method=None).fillna(0)
    elif change_type == 'delta':
        # Delta differences
        changes = pd.Series(data).diff().fillna(0)
    else:
        print('Wrong Change Type!')
    
    # Create sequences
    sequences = []
    for i in range(0, len(changes) - seq_length, stride):
        sequence = changes[i:i + seq_length]
        # Only include sequences with no missing values
        if not np.isnan(sequence).any():
            sequences.append(sequence)
    
    return np.array(sequences)

def tokenize_deltas(data, bin_start, bin_stop, bin_width):
    
    # Assuming data is a NumPy array of shape (n_sequences, seq_length)
    n_sequences, _ = data.shape

    # Bin range and Bin width
    num_bins = int((bin_stop - bin_start) / bin_width)  # Calculate number of bins
    bin_edges = np.arange(bin_start, bin_stop + bin_width, bin_width)

    # Tokenize each sequence individually (no flattening)
    tokenized_sequences = np.empty_like(data)
    for i in range(n_sequences):
        tokenized_data = np.digitize(data[i], bins=bin_edges, right=False) - 1
        tokenized_sequences[i] = np.clip(tokenized_data, 0, num_bins - 1)

    return tokenized_sequences

def to_onehot(tokens, vocab_size):
    batch_size, seq_length = tokens.shape
    onehot = th.zeros(batch_size, seq_length, vocab_size, device=device)  # Add device parameter
    return onehot.scatter_(2, tokens.unsqueeze(-1), 1)

def to_onehot_inference(tokens, vocab_size):
    onehot = th.zeros(1, 1, vocab_size, device=device)  # Add device parameter
    return onehot.scatter_(2, tokens.view(1, 1, 1), 1)

power_data = pd.read_csv('processed_power_data.csv')
delta_data = create_sequences(power_data['Global_active_power'].values, seq_length=52, stride=20, change_type='delta')
token_limit = 2

bin_width = np.std(np.concatenate(delta_data)) / 32
bin_start=-token_limit
bin_stop=token_limit
token_size = int((bin_stop - bin_start) / bin_width)  # Calculate number of bins

token_data = tokenize_deltas(delta_data, bin_start=bin_start, bin_stop=bin_stop, bin_width=bin_width)

n_sequences = delta_data.shape[0]
n_days = delta_data.shape[1] 

train_ratio = 0.75
# Chronological splitting
train_size = int(train_ratio * n_sequences)
val_size = int((1-train_ratio)/2 * n_sequences)

# Split maintaining temporal order
train_set = delta_data[:train_size]
val_set = delta_data[train_size:train_size+val_size]
test_set = delta_data[train_size+val_size:]

X_train = tokenize_deltas(train_set, bin_start=bin_start, bin_stop=bin_stop, bin_width=bin_width)
X_val = tokenize_deltas(val_set, bin_start=bin_start, bin_stop=bin_stop, bin_width=bin_width)
X_test = tokenize_deltas(test_set, bin_start=bin_start, bin_stop=bin_stop, bin_width=bin_width)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# Save everything needed for training
data_dict = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'token_size': token_size,
    'token_limit': token_limit,
    'bin_start': bin_start,
    'bin_stop': bin_stop,
    'bin_width': bin_width
}

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print("Data saved with shapes:")
print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)