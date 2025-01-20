import ray
from pathlib import Path
import pickle
from models import *
from visualization import *
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO

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
sw_posttraining = 'sw_posttrain_50.pth'
gail_training = "GAIL_weights_50_delta_bce.zip"

# Define base configuration
base_config = {
    
    'timesteps': 200000,
    'n_steps': (n_days-2)*100,
    'batch_size': (n_days-2)*10,
    'n_epochs': 4,

    'lr_gail': 1e-06,
    
    'ent_coef': 0.001,
    'clip_range': 0.03,
    'vf_coef': 0.5,
    'gamma': 0.995,
    'gae_lambda': 0.95,
    
    'hidden_dim': 128,
    'hidden_layers': 2
}

seeds = range(1000, 1020)  # 20 seeds

def get_experiment_dir(seed, base_dir="results_20seeds"):
    """Create directory name based on seed"""
    return Path(base_dir) / f"seed_{seed}"

def run_single_experiment(config, seed, save_dir):
    """Your current training logic goes here"""
    # Set random seed
    th.manual_seed(seed)
    np.random.seed(seed)

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup discriminator and environment
    discriminator = LSTM_Discriminator(vocab_size=token_size, embedding_dim=64, hidden_dim=256)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    env_GAIL = CustomEnv(token_size, bin_start, bin_stop, bin_width,
                     X_train, X_val,
                     discriminator, d_optimizer)
    env_GAIL = Monitor(env_GAIL, filename=None)

    # Merge base config with experiment config
    full_config = {**base_config, **config}
    
    # Create model with config
    model_pre_weight = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env_GAIL,
        verbose=0,
        n_steps=full_config['n_steps'],
        batch_size=full_config['batch_size'],
        n_epochs=full_config['n_epochs'],
        clip_range=full_config['clip_range'],
        ent_coef=full_config['ent_coef'],
        vf_coef=full_config['vf_coef'],
        gamma=full_config['gamma'],
        gae_lambda=full_config['gae_lambda'],
        learning_rate=full_config['lr_gail'],
        policy_kwargs=dict(
            lstm_hidden_size=full_config['hidden_dim'],
            n_lstm_layers=full_config['hidden_layers'],
            net_arch=dict(pi=[], vf=[]),
            shared_lstm=True,
            enable_critic_lstm=False,
        )
    )

    # Your existing callback and training code
    callback_GAIL = CustomCallback(
        verbose=0, 
        display_rollout=False, 
        disc_batch_size=128, 
        gail_training=str(save_dir / "best_model.zip")
    )
    
    model_post_weight = transfer_weights_from_saved(
        sw_pretraining, 
        model_pre_weight, 
        True, 
        token_size, 
        full_config['hidden_dim'], 
        token_size, 
        full_config['hidden_layers']
    )

    # Train
    model_post_weight.learn(full_config['timesteps'], callback=callback_GAIL)
    env_GAIL.close()

    # Save results
    results = {
        'config': full_config,
        'seed': seed,
        'metrics': {
            'pg_losses': callback_GAIL.pg_losses,
            'value_losses': callback_GAIL.value_losses,
            'entropy_losses': callback_GAIL.entropy_losses,
            'rewards': callback_GAIL.rewards,
            'wasserstein': callback_GAIL.sequence_metrics['wasserstein'],
            'kl_div': callback_GAIL.sequence_metrics['kl_div'],
            'disc_loss': callback_GAIL.disc_metrics_per_batch['loss'],
            'disc_acc': callback_GAIL.disc_metrics_per_batch['accuracy'],
            'disc_acc_diff': callback_GAIL.disc_metrics_per_batch['accuracy_difference']
        }
    }
    
    with open(save_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results



@ray.remote
class ExperimentWorker:
    def __init__(self):
        pass
        
    def run_experiment(self, config, seed, save_dir):
        return run_single_experiment(config, seed, save_dir)

if __name__ == "__main__":
    ray.init(num_cpus=4)
    workers = [ExperimentWorker.remote() for _ in range(4)]
    
    # Create experiment queue with just seeds
    experiments = [(base_config, seed) for seed in seeds]
    
    # Run experiments in batches
    results = []
    for i in range(0, len(experiments), 4):
        batch = experiments[i:i+4]
        batch_futures = [
            workers[j].run_experiment.remote(
                config=exp[0],
                seed=exp[1],
                save_dir=get_experiment_dir(exp[1])  # Changed to new directory function
            )
            for j, exp in enumerate(batch)
        ]
        batch_results = ray.get(batch_futures)
        results.extend(batch_results)
        print(f"Completed runs {i+1}-{min(i+4, len(experiments))} of {len(experiments)}")

    # Save final combined results
    with open('all_20seeds_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        