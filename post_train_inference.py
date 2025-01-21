import pickle
from models import *
from data_processing import to_onehot, to_onehot_inference
from torch.nn import functional as F
import ray
from pathlib import Path
import torch as th

# Load the processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)


# Extract what you need
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
token_size = data['token_size']
token_limit = data['token_limit']
bin_start = data['bin_start']
bin_stop = data['bin_stop']
bin_width = data['bin_width']

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

n_days = X_train.shape[1]

sw_pretraining = 'sw_pretrain_50.pth'

# Rest of your parameters remain the same
hidden_dim = 128
batch_size = 256
learning_rate = 2e-4
seq_len = n_days-1
num_epochs = 50
hidden_layers = 2
val_loss_pretrain = 2.5018


@ray.remote
class PostTrainingWorker:
    
    def __init__(self):
        pass

    def run_post_training(self, seed, save_dir):
        # Set random seed
        th.manual_seed(seed)
        np.random.seed(seed)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Model initialization
        post_supervised_model = LSTMModel(input_dim=token_size, hidden_dim=hidden_dim, output_dim=token_size, num_layers=hidden_layers)
        post_supervised_optimizer = th.optim.Adam(post_supervised_model.parameters(), lr=learning_rate)
        
        # Load pretrained weights
        checkpoint = th.load(sw_pretraining)
        post_supervised_model.load_state_dict(checkpoint['model_state_dict'])
        post_supervised_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        post_supervised_criterion = nn.CrossEntropyLoss()
        
        train_loss = []
        val_loss = []
        best_val_loss = val_loss_pretrain
        max_grad_norm = 1.0

        # Pre-allocate tensors for validation
        val_input_tokens = th.tensor(X_val[:, :-1], dtype=th.long)
        val_targets = th.tensor(X_val[:, 1:], dtype=th.long)
        val_onehot = to_onehot(val_input_tokens, token_size)
        train_tokens = th.tensor(X_train, dtype=th.long)

        # Your training loop
        for epoch in range(num_epochs):
            temp_train_loss = []
            temp_val_loss = []
            
            # Training
            post_supervised_model.train()
            indices = th.randperm(X_train.shape[0])

            for batch_idx in range(0, X_train.shape[0], batch_size):
                # Your existing batch training code...
                batch_indices = indices[batch_idx:batch_idx + batch_size]
                batch = train_tokens[batch_indices]

                input_tokens = batch[:, :-1]
                inputs = to_onehot(input_tokens, token_size)
                targets = batch[:, 1:]

                logits, _ = post_supervised_model(inputs, None)
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)

                post_supervised_optimizer.zero_grad(set_to_none=True)
                loss = post_supervised_criterion(logits, targets)
                loss.backward()

                th.nn.utils.clip_grad_norm_(post_supervised_model.parameters(), max_grad_norm)
                post_supervised_optimizer.step()
                
                temp_train_loss.append(loss.item())
            
            # Validation
            post_supervised_model.eval()
            with th.no_grad():
                logits, _ = post_supervised_model(val_onehot, None)
                logits = logits.reshape(-1, logits.size(-1))
                targets = val_targets.reshape(-1)
                loss = post_supervised_criterion(logits, targets)
                temp_val_loss.append(loss.item())

            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                checkpoint = {
                    'model_state_dict': post_supervised_model.state_dict(),
                    'optimizer_state_dict': post_supervised_optimizer.state_dict()
                }
                th.save(checkpoint, save_dir / "best_model.pth")

            train_loss.extend(temp_train_loss)
            val_loss.append(loss.item())
            print(f"Seed {seed} - Epoch {epoch+1}/{num_epochs} - Train Loss: {np.mean(temp_train_loss):.4f} - Val Loss: {loss.item():.4f}")

        
        if best_val_loss < val_loss_pretrain:  # If we found a better model during post-training
            print(f"Seed {seed} - Using post-trained model for inference (val_loss: {best_val_loss:.4f})")
            inference_model = LSTMModel(input_dim=token_size, hidden_dim=hidden_dim, output_dim=token_size)
            best_checkpoint = th.load(save_dir / "best_model.pth")
            inference_model.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            print(f"Seed {seed} - Using original pretrained model for inference")
            inference_model = LSTMModel(input_dim=token_size, hidden_dim=hidden_dim, output_dim=token_size)
            original_checkpoint = th.load(sw_pretraining)
            inference_model.load_state_dict(original_checkpoint['model_state_dict'])

        # Run inference
        supervised_post_data = run_inference(post_supervised_model, X_test)
        
        # Save results
        results = {
            'seed': seed,
            'train_losses': train_loss,
            'val_losses': val_loss,
            'inference_data': supervised_post_data
        }
        
        with open(save_dir / 'post_supervised_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        return results

def run_inference(model, test_data):
    test_data_inference = th.tensor(test_data, dtype=th.long)
    supervised_post_data = []
    
    model.eval()
    with th.no_grad():
        for sequence in test_data_inference:
            init_real_token = sequence[0]
            sim_trajectory = [init_real_token.item()]
            hidden = None
            last_token = th.zeros(1, dtype=th.long)
            
            for i in range(1, len(sequence)):
                last_token[0] = sim_trajectory[-1]
                input_onehot = to_onehot_inference(last_token, token_size)
                logits, hidden = model(input_onehot, hidden)
                probs = F.softmax(logits, dim=-1).squeeze()
                predicted_token = th.multinomial(probs, num_samples=1).item()
                sim_trajectory.append(predicted_token)
            
            supervised_post_data.append(sim_trajectory[1:])
    
    return np.array(supervised_post_data)

def run_post_training_parallel():
    # Initialize Ray
    ray.init(num_cpus=4)
    
    # Create workers
    workers = [PostTrainingWorker.remote() for _ in range(4)]
    
    # Create experiment queue
    seeds = range(1000, 1030)  # 30 seeds
    
    # Run experiments in batches
    results = []
    for i in range(0, len(seeds), 4):
        batch_seeds = seeds[i:i+4]
        batch_futures = [
            workers[j].run_post_training.remote(
                seed=seed,
                save_dir=f"post_supervised_results/seed_{seed}"
            )
            for j, seed in enumerate(batch_seeds)
        ]
        batch_results = ray.get(batch_futures)
        results.extend(batch_results)
        print(f"Completed post-training batch {i//4 + 1}/{len(seeds)//4}")
    
    # Save combined results
    with open('all_post_supervised_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    ray.shutdown()
    return results

if __name__ == "__main__":
    results = run_post_training_parallel()

