# Same imports as data_prep.py
# import numpy as np
# import torch as th
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import os
# from collections import Counter
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import acf
# from scipy.stats import wasserstein_distance, gaussian_kde
# import seaborn as sns
# from scipy.stats import skew, kurtosis
# import gymnasium as gym
# from gymnasium import spaces
# from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.monitor import Monitor

from data_processing import *
from models import *
from visualization import *



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