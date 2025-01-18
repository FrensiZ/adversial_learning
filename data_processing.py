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
    """Convert batch of tokens to one-hot vectors using efficient PyTorch operations"""
    batch_size, seq_length = tokens.shape
    # Create indices tensor
    onehot = th.zeros(batch_size, seq_length, vocab_size, device=tokens.device)
    # Use scatter_ for efficient one-hot encoding
    return onehot.scatter_(2, tokens.unsqueeze(-1), 1)

def to_onehot_inference(tokens, vocab_size):
    """Efficient one-hot encoding using scatter"""
    onehot = th.zeros(1, 1, vocab_size)
    return onehot.scatter_(2, tokens.view(1, 1, 1), 1)
