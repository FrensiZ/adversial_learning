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
    'n_steps': (n_days-2)*100,
    'batch_size': (n_days-2)*10,
    'n_epochs': 4,
    'timesteps': 150000,  # Reduced from 300000
    'vf_coef': 0.3,
    'gamma': 1,
    'gae_lambda': 0.95,
    'hidden_dim': 128,
    'hidden_layers': 2
}

# Define experiment configurations
experiment_configs = [
    {
        'lr_gail': lr,
        'ent_coef': ent,
        'clip_range': clip,
    }
    for lr in [1e-7, 5e-7, 1e-6]
    for ent in [0.001, 0.005, 0.01]
    for clip in [0.02, 0.05, 0.1]
]

seeds = [1000, 1001, 1002]

def get_experiment_dir(config, seed, base_dir="results"):
    """Create unique directory name based on config and seed"""
    config_str = f"lr{config['lr_gail']}_ent{config['ent_coef']}_clip{config['clip_range']}"
    return Path(base_dir) / config_str / f"seed_{seed}"

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
    # Initialize Ray
    ray.init(num_cpus=4)
    
    # Create workers
    workers = [ExperimentWorker.remote() for _ in range(4)]
    
    # Create experiment queue
    experiments = [
        (config, seed) 
        for config in experiment_configs 
        for seed in seeds
    ]
    
    # Run experiments in batches
    results = []
    for i in range(0, len(experiments), 4):
        batch = experiments[i:i+4]
        batch_futures = [
            workers[j].run_experiment.remote(
                config=exp[0],
                seed=exp[1],
                save_dir=get_experiment_dir(exp[0], exp[1])
            )
            for j, exp in enumerate(batch)
        ]
        batch_results = ray.get(batch_futures)
        results.extend(batch_results)
        print(f"Completed batch {i//4 + 1}/{len(experiments)//4}")

    # Save final combined results
    with open('all_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        