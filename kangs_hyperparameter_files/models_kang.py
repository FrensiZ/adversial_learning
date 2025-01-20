import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from scipy.stats import wasserstein_distance, gaussian_kde
from collections import Counter

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.to(device)  # Add this line
    
    def forward(self, x, hidden):
        # x shape now: [batch_size, sequence_length, vocab_size]
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        logits = self.fc(lstm_out)
        return logits, (h_n, c_n)


class LSTM_Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM_Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.to(device)  # Add this line
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        logits = self.fc(lstm_out)
        logits = logits.squeeze(-1)
        return logits, hidden
    
class CustomEnv(gym.Env):
    
    def __init__(self,
                 token_size, bin_start, bin_stop, bin_width,
                 train_data, val_data,
                 discriminator, d_optimizer):
        
        super(CustomEnv, self).__init__()

        # Spaces
        self.action_space = spaces.Discrete(token_size)        # Action Space
        self.observation_space = spaces.Discrete(token_size)   # Observation Spacepass

        self.num_bins = int((bin_stop - bin_start) / bin_width)
        self.bin_edges = np.arange(bin_start, bin_stop + bin_width, bin_width)

        # Data
        self.train_data = train_data
        self.val_data = val_data
        
        # Discriminator
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.criterion = nn.BCEWithLogitsLoss()

        # Reset
        self.seq_len = None
        self.curr_seq = []
        self.seq_position = 0
        self.hidden_states = None

        # ### HERE
        # self.last_bce = -np.log(0.5)
        # ### HERE
    
    def step(self, action):

        assert self.action_space.contains(action), "Invalid Action"

        self.seq_position += 1

        self.curr_seq.append(action)

        ###=====REWARD=====###
        reward = self._get_reward(action)
        ###=====REWARD=====###
        
        done = self._is_done()
        
        next_obs = action

        return next_obs, reward, done, False, {}        # observation, reward, done, truncated, info
    
    def reset(self, *, seed = None, options = None):

        seq_idx = np.random.choice(self.train_data.shape[0], size = 1).item()
        real_seq = self.train_data[seq_idx]
        
        # Reset
        self.seq_len = len(real_seq)
        self.curr_seq = []
        self.seq_position = 0
        self.hidden_states = None
        
        start_token = int(real_seq[0])
        
        # ### HERE
        # self.last_bce = -np.log(0.5)
        # ### HERE
        
        return start_token, {}
    
    def _get_reward(self, action):

        self.discriminator.eval()

        with th.no_grad():
            temp_action = th.tensor(np.array([[action]]), dtype=th.long).to(device)  # Add .to(device)
            prediction, self.hidden_states = self.discriminator(temp_action, self.hidden_states)
            prediction = prediction[0,0]
            label = th.ones_like(prediction)  # This will inherit device from prediction
            seq_loss = self.criterion(prediction, label)
            reward = -(seq_loss.item())
            return reward


    def _is_done(self):

        return self.seq_position == (self.seq_len - 1)

    def get_train_sequence(self):
        seq_idx = np.random.choice(self.train_data.shape[0], size = 1).item()
        real_seq = self.train_data[seq_idx]
        return real_seq
    
    def get_val_sequence(self):
        seq_idx = np.random.choice(self.val_data.shape[0], size = 1).item()
        real_seq = self.val_data[seq_idx]
        return real_seq

    def discriminator_accuracy(self, real_data, fake_data):
        
        self.discriminator.eval()
        
        with th.no_grad():
            batch_real = th.tensor(real_data, dtype=th.long).to(device)  # Add .to(device)
            batch_fake = th.tensor(fake_data, dtype=th.long).to(device)  # Add .to(device)
            
            seq_len = batch_real.size(1)
            split_idx = seq_len // 2
            
            # Full sequence predictions
            real_preds, _ = self.discriminator(batch_real, None)
            fake_preds, _ = self.discriminator(batch_fake, None)
            
            real_probs = th.sigmoid(real_preds)
            fake_probs = th.sigmoid(fake_preds)
            
            # Full accuracy
            correct_real = (real_probs > 0.5).float().sum()
            correct_fake = (fake_probs < 0.5).float().sum()
            total = real_probs.numel() + fake_probs.numel()
            acc_full = (correct_real + correct_fake) / total
            
            # First half accuracy
            correct_real_first = (real_probs[:split_idx] > 0.5).float().sum()
            correct_fake_first = (fake_probs[:split_idx] < 0.5).float().sum()
            total_first = real_probs[:split_idx].numel() + fake_probs[:split_idx].numel()
            acc_1 = (correct_real_first + correct_fake_first) / total_first
            
            # Second half accuracy
            correct_real_second = (real_probs[split_idx:] > 0.5).float().sum()
            correct_fake_second = (fake_probs[split_idx:] < 0.5).float().sum()
            total_second = real_probs[split_idx:].numel() + fake_probs[split_idx:].numel()
            acc_2 = (correct_real_second + correct_fake_second) / total_second
            
            return acc_full.item(), acc_1.item(), acc_2.item(), (acc_2 - acc_1).item()
    
    def train_discriminator(self, batch_real, batch_fake):
        
        self.discriminator.train()
        
        batch_real = th.tensor(batch_real, dtype=th.long).to(device)  # Add .to(device)
        batch_fake = th.tensor(batch_fake, dtype=th.long).to(device)  # Add .to(device)

        # Forward pass - real data
        real_preds, _ = self.discriminator(batch_real, None)
        real_preds = real_preds.view(-1)
        real_labels = th.ones_like(real_preds) * 0.9  # Label smoothing
        real_loss = self.criterion(real_preds, real_labels)

        # Forward pass - fake data
        fake_preds, _ = self.discriminator(batch_fake, None)
        fake_preds = fake_preds.view(-1)
        fake_labels = th.ones_like(fake_preds) * 0.1  # Label smoothing
        fake_loss = self.criterion(fake_preds, fake_labels)

        # Combined loss and update
        d_loss = (real_loss + fake_loss) / 2
        self.d_optimizer.zero_grad()
        d_loss.backward()
        th.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)  # Add gradient clipping
        self.d_optimizer.step()

        with th.no_grad():
            discriminator_loss = d_loss.item()
        
        return discriminator_loss

class CustomCallback(BaseCallback):

    def __init__(self, verbose, display_rollout, disc_batch_size, gail_training):
        
        super(CustomCallback, self).__init__(verbose)

        self.val_size = None
        
        # Display Rollout
        self.display_rollout = display_rollout
        # Losses
        self.pg_losses, self.value_losses, self.entropy_losses = [], [], []
        # Rewards
        self.rewards = []
        
        self.batch_size = disc_batch_size

        # KL and Wasserstein
        self.sequence_metrics = {
            'kl_div': [],
            'wasserstein': []
        }

        # Metrics after every network update of the discriminator
        self.disc_metrics_per_batch = {
            'loss': [], 
            'accuracy': [],
            'accuracy_1': [],
            'accuracy_2': [],
            'accuracy_difference': [],
            'n_batches': []
        }

        # Save best model
        self.best_wasserstein = float('inf')  # Initialize with infinity
        self.best_model_path = gail_training   # Path to save best model

    def _on_training_start(self) -> None:
        
        self.val_size = self.training_env.envs[0].unwrapped.val_data.shape[0]    

    def _on_rollout_start(self) -> None:
        
        self.save_losses()

        real_data, fake_data = self.get_sequences(n = self.val_size, train = False)

        wasser_dist = self.calculate_wasserstein_distance(real_data, fake_data)
        kl_div = self.calculate_kl_divergence(real_data, fake_data)

        self.sequence_metrics['wasserstein'].append(wasser_dist)
        self.sequence_metrics['kl_div'].append(kl_div)

        # Save model if we have a new best Wasserstein distance
        if wasser_dist < self.best_wasserstein:
            self.best_wasserstein = wasser_dist
            print(f"\nNew best Wasserstein distance: {wasser_dist:.3f}, saving model...")
            self.model.save(f"{self.best_model_path}")

        for env in self.training_env.envs:

            pre_acc, _, _, pre_diff = env.unwrapped.discriminator_accuracy(real_data, fake_data)

            metrics = {
                'losses': [], 
                'accuracies': [],
                'pre_acc': 0,
                'post_acc': 0
                }

            max_accuracy = 0.60     # Don't want discriminator too strong
            min_batches = 0         # Changed from 10
            max_batches = 50        # Changed from 100

            # Dynamic batch sizing
            curr_batch_size = self.batch_size
            if pre_acc < 0.52:  # If discriminator is very weak
                curr_batch_size = self.batch_size * 2  # Train on more data
            
            for batch in range(max_batches):
                
                real_batch, fake_batch = self.get_sequences(n=curr_batch_size, train=True)
                batch_loss = env.unwrapped.train_discriminator(real_batch, fake_batch)
                curr_acc, curr_acc_1, curr_acc_2, curr_acc_diff = env.unwrapped.discriminator_accuracy(real_batch, fake_batch)

                metrics['losses'].append(batch_loss)
                metrics['accuracies'].append(curr_acc)
                
                self.disc_metrics_per_batch['loss'].append(batch_loss)
                self.disc_metrics_per_batch['accuracy'].append(curr_acc)
                self.disc_metrics_per_batch['accuracy_1'].append(curr_acc_1)
                self.disc_metrics_per_batch['accuracy_2'].append(curr_acc_2)
                self.disc_metrics_per_batch['accuracy_difference'].append(curr_acc_diff)

                if batch >= min_batches:
                    recent_losses = metrics['losses'][-3:]
                    loss_mean = np.mean(recent_losses)
                    ideal_loss = -np.log(0.5)  # Binary cross entropy at p=0.5
                    
                    # Your existing conditions
                    if curr_acc > max_accuracy:
                        print(f"Discriminator too strong ({curr_acc:.3f}), stopping")
                        break
                    
                    recent_acc = np.mean(metrics['accuracies'][-3:])
                    if 0.46 < recent_acc < 0.54 and np.std(metrics['accuracies'][-3:]) < 0.01:
                        print(f"Discriminator balanced near 0.5 ({recent_acc:.3f}), stopping")
                        break
                        
                    # Add this new condition
                    if abs(loss_mean - ideal_loss) < 0.1:
                        print(f"Loss stabilized near ideal value ({loss_mean:.3f}), stopping")
                        break

                    # This condition might be redundant now
                    if batch > 10 and np.std(metrics['accuracies'][-10:]) < 0.01:
                        print("Discriminator converged, stopping")
                        break
                                
                # Adjust batch size if needed
                if batch % 20 == 0:  # Every 10 batches
                    if curr_acc < 0.52:  # Changed from 0.60
                        curr_batch_size = min(curr_batch_size * 2, 256)
                    elif curr_acc > 0.65:  # Changed from 0.80
                        curr_batch_size = max(curr_batch_size // 2, self.batch_size)
                
            self.disc_metrics_per_batch['n_batches'].append(batch + 1)
            
            post_acc, _, _, post_diff = env.unwrapped.discriminator_accuracy(real_data, fake_data)

            print(f"Wasserstein: {wasser_dist:.3f} - KL Div: {kl_div:.3f} - Stopped after {batch + 1} batches")
            print(f"Pre Accuracy: {pre_acc:.3f} - Post Accuracy: {post_acc:.3f} - Pre Acc Diff: {pre_diff:.3f} - Post Acc Diff: {post_diff:.3f}")
    
    def _on_step(self) -> bool:

        return True

    def _on_rollout_end(self) -> None:

        if self.display_rollout:
            self.display_rollout_buffer()
        
        self.save_reward()
    
    def _on_training_end(self) -> None:
        
        self.save_losses()

    def save_losses(self) -> None:
        self.pg_losses.append(self.logger.name_to_value.get('train/policy_gradient_loss', None))
        self.value_losses.append(self.logger.name_to_value.get('train/value_loss', None))
        self.entropy_losses.append(self.logger.name_to_value.get('train/entropy_loss', None))

    def save_reward(self):
        # Get rewards without reshaping
        rewards = self.model.rollout_buffer.rewards  # Shape: (n_steps, n_envs)
        episode_starts = self.model.rollout_buffer.episode_starts
        
        all_sequence_rewards = []
        
        # Process each environment separately
        for env_idx in range(self.training_env.num_envs):
            # Get rewards and episode starts for this environment
            env_rewards = rewards[:, env_idx]
            env_starts = episode_starts[:, env_idx]
            
            # Find episode starts for this env
            ep_start_idx = np.where(env_starts)[0]
            if not env_starts[0]:
                ep_start_idx = np.r_[0, ep_start_idx]
                
            # Calculate rewards for this env's sequences
            sequence_rewards = np.add.reduceat(env_rewards, np.r_[0, ep_start_idx[1:]])
            all_sequence_rewards.extend(sequence_rewards)
        
        # Store average across all envs
        self.rewards.append(np.mean(all_sequence_rewards))

    def display_rollout_buffer(self):
        print("\nRollout Buffer Contents:")
        
        # Get original shaped arrays
        observations = self.model.rollout_buffer.observations  # Shape: (n_steps, n_envs)
        actions = self.model.rollout_buffer.actions
        rewards = self.model.rollout_buffer.rewards
        returns = self.model.rollout_buffer.returns
        episode_starts = self.model.rollout_buffer.episode_starts
        values = self.model.rollout_buffer.values
        log_probs = self.model.rollout_buffer.log_probs
        advantages = self.model.rollout_buffer.advantages
        
        # Print header with line
        header = "  env_id    step    observations  actions  rewards  returns  episode_starts  values  log_probs  advantages"
        print("-" * (len(header) + 5))
        print(header)
        print("-" * (len(header) + 5))
        
        # For each environment
        for env_idx in range(self.training_env.num_envs):
            # Get data for this environment
            env_obs = observations[:, env_idx]
            env_actions = actions[:, env_idx]
            env_rewards = rewards[:, env_idx]
            env_returns = returns[:, env_idx]
            env_starts = episode_starts[:, env_idx]
            env_values = values[:, env_idx]
            env_log_probs = log_probs[:, env_idx]
            env_advantages = advantages[:, env_idx]
            
            # Find episode starts for this env
            ep_start_idx = np.where(env_starts)[0]
            
            # Print rows for this environment
            last_idx = -1
            for step in range(len(env_obs)):
                if step in ep_start_idx:
                    if last_idx != -1:
                        print("=" * (len(header) + 5))
                    print(f"Environment {env_idx} - New Episode")
                    print("-" * (len(header) + 5))
                
                # Convert arrays to strings or get first element if array
                obs_str = str(env_obs[step].item()) if hasattr(env_obs[step], 'item') else str(env_obs[step])
                action_str = str(env_actions[step].item()) if hasattr(env_actions[step], 'item') else str(env_actions[step])
                
                print(
                    f"Obs: {obs_str:8s}  "
                    f"Act: {action_str:4s}  "
                    f"Rew: {env_rewards[step]:6.2f}  "
                    f"Ret: {env_returns[step]:7.2f}  "
                    f"Start: {int(env_starts[step]):1d}  "
                    f"Val: {env_values[step]:6.2f}  "
                    f"LogP: {env_log_probs[step]:8.2f}  "
                    f"Adv: {env_advantages[step]:8.2f}")
                
                last_idx = step
            
            print("=" * (len(header) + 5))
            print(f"End of Environment {env_idx}")
            print("\n")

    def get_sequences(self, n, train):

        real_data = []
        fake_data = []
        
        # 1. Batch process for each environment
        for env in self.training_env.envs:
            # Get all real sequences at once
            if train:
                real_seqs = [env.unwrapped.get_train_sequence() for _ in range(n)]
            else:
                real_seqs = [env.unwrapped.get_val_sequence() for _ in range(n)]
            
            # Prepare batch for model.predict
            batch_size = n
            actions = np.array([seq[0] for seq in real_seqs])
            lstm_states = None
            episode_starts = np.ones((batch_size,), dtype=bool)
            
            # Initialize all fake sequences
            curr_seqs = [[] for _ in range(batch_size)]
            seq_len = len(real_seqs[0]) - 1
            
            # Generate all sequences in parallel
            for _ in range(seq_len):
                # Predict entire batch at once
                actions, lstm_states = self.model.predict(
                    actions,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=False
                )
                
                # Update all sequences at once
                for i, action in enumerate(actions):
                    curr_seqs[i].append(action.item())
                
                episode_starts = np.zeros((batch_size,), dtype=bool)
            
            # Append results
            fake_data.extend(curr_seqs)
            real_data.extend([seq[1:] for seq in real_seqs])
        
        return np.array(real_data), np.array(fake_data)
    
    def calculate_wasserstein_distance(self, real_data, fake_data):
        
        wasserstein_distances = []
        for i in range(len(real_data[0])):
            w_dist = wasserstein_distance(real_data[:, i], fake_data[:, i])
            wasserstein_distances.append(w_dist)

        # Different ways to analyze the results
        return np.mean(wasserstein_distances)  # Overall performance
    
    def calculate_kl_divergence(self, real_sequences, fake_sequences, eps=1e-10):

        # Flatten sequences and count tokens
        real_tokens = np.concatenate(real_sequences)
        fake_tokens = np.concatenate(fake_sequences)
        
        # Get unique vocabulary from both distributions
        vocab = set(real_tokens) | set(fake_tokens)
        
        # Calculate token counts and totals
        real_counts = Counter(real_tokens)
        fake_counts = Counter(fake_tokens)
        
        total_real = len(real_tokens)
        total_fake = len(fake_tokens)
        
        # Calculate probabilities with smoothing for all tokens in vocabulary
        real_probs = {token: (real_counts.get(token, 0) + eps) / (total_real + eps * len(vocab))
                    for token in vocab}
        fake_probs = {token: (fake_counts.get(token, 0) + eps) / (total_fake + eps * len(vocab))
                    for token in vocab}
        
        # Calculate KL divergence
        kl_div = sum(p_real * np.log2(p_real / fake_probs[token])
                    for token, p_real in real_probs.items())
        
        return kl_div


def transfer_weights_from_saved(weights_path, ppo_model, transfer_head, input_dim, hidden_dim, output_dim, num_layers):
    """
    Load saved supervised weights and transfer to PPO model with optional head transfer
    
    Args:
        weights_path: Path to the saved supervised model weights
        ppo_model: The PPO model to transfer weights to
        transfer_head: If True, also transfer fc layer weights to action_net
        input_dim: Input dimension of the supervised model
        hidden_dim: Hidden dimension of the supervised model
        output_dim: Output dimension of the supervised model
    """
    # Create temporary supervised model to load weights into
    temp_supervised = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    
    # Load the saved weights
    saved_weights = th.load(weights_path, weights_only=False)
    temp_supervised.load_state_dict(saved_weights['model_state_dict'])
    
    
    # Transfer LSTM weights
    print("\n=== Transferring LSTM Weights ===")
    supervised_state_dict = temp_supervised.state_dict()
    ppo_lstm_dict = ppo_model.policy.lstm_actor.state_dict()
    
    # Print shapes before transfer for verification
    print("\nWeight shapes before transfer:")
    print("\nSupervised LSTM weights:")
    for key, value in supervised_state_dict.items():
        if 'lstm' in key:
            print(f"{key}: {value.shape}")
    
    print("\nPPO LSTM weights:")
    for key, value in ppo_lstm_dict.items():
        print(f"{key}: {value.shape}")
    
    # Transfer LSTM weights
    lstm_transfer_count = 0
    for ppo_key in ppo_lstm_dict.keys():
        supervised_key = f"lstm.{ppo_key}"
        if supervised_key in supervised_state_dict:
            if ppo_lstm_dict[ppo_key].shape == supervised_state_dict[supervised_key].shape:
                ppo_lstm_dict[ppo_key].copy_(supervised_state_dict[supervised_key])
                lstm_transfer_count += 1
                print(f"Transferred weights for {ppo_key}")
            else:
                print(f"Shape mismatch for {ppo_key}")
    
    # Load the LSTM weights
    ppo_model.policy.lstm_actor.load_state_dict(ppo_lstm_dict)
    print(f"\nSuccessfully transferred {lstm_transfer_count} LSTM weight tensors")
    
    # Transfer head weights if requested
    if transfer_head:
        print("\n=== Transferring Head Weights ===")
        # Get supervised fc weights and biases
        fc_weight = supervised_state_dict['fc.weight']
        fc_bias = supervised_state_dict['fc.bias']
        
        # Get PPO action_net weights and biases
        action_net_state_dict = ppo_model.policy.action_net.state_dict()
        
        print("\nHead weight shapes:")
        print(f"Supervised fc weight: {fc_weight.shape}")
        print(f"Supervised fc bias: {fc_bias.shape}")
        print(f"PPO action_net weight: {action_net_state_dict['weight'].shape}")
        print(f"PPO action_net bias: {action_net_state_dict['bias'].shape}")
        
        # Verify shapes match before transfer
        if (fc_weight.shape == action_net_state_dict['weight'].shape and 
            fc_bias.shape == action_net_state_dict['bias'].shape):
            # Transfer weights
            action_net_state_dict['weight'].copy_(fc_weight)
            action_net_state_dict['bias'].copy_(fc_bias)
            ppo_model.policy.action_net.load_state_dict(action_net_state_dict)
            print("Successfully transferred head weights")
        else:
            print("Shape mismatch in head weights - transfer aborted")
    
    return ppo_model