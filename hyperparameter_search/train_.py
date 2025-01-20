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

# LSTM Settings
hidden_dim = 128
#batch_size = 256
#lr_supervised = 2e-4
#seq_len = n_days-1
#num_epochs = 20
hidden_layers = 2

sw_pretraining = 'sw_pretrain_50.pth'
sw_posttraining = 'sw_posttrain_50.pth'
gail_training = "GAIL_weights_50_delta_bce.zip"

discriminator = LSTM_Discriminator(vocab_size=token_size, embedding_dim=64, hidden_dim=256)
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

env_GAIL = CustomEnv(token_size, bin_start, bin_stop, bin_width,
                 X_train, X_val,
                 discriminator, d_optimizer)

env_GAIL = Monitor(env_GAIL, filename=None)


##### GAIL Settings
##### GAIL Settings

n_steps=(n_days-2)*100
batch_size=(n_days-2)*10
n_epochs = 4
timesteps = 300000

clip_range=0.05            # Lower PPO clipping
ent_coef=0.005             # Moderate exploration
vf_coef=0.3                # Balanced value function importance
gamma=1                    # Standard discount
gae_lambda=0.95            # Standard GAE lambda
lr_gail=5e-7               # Just small changes because performance is already perfect.

##### GAIL Settings
##### GAIL Settings


model_pre_weight = RecurrentPPO(

    policy='MlpLstmPolicy',
    env=env_GAIL,
    verbose=0,

    n_steps=n_steps,            # Fewer steps per update
    batch_size=batch_size,      # Smaller batches
    n_epochs=n_epochs,          # Fewer epochs (less aggressive updates)
    
    clip_range=clip_range,            # Lower PPO clipping
    ent_coef=ent_coef,             # Moderate exploration
    vf_coef=vf_coef,                # Balanced value function importance
    gamma=gamma,                    # Standard discount
    gae_lambda=gae_lambda,            # Standard GAE lambda

    learning_rate=lr_gail,         # Just small changes because performance is already perfect.
    
    policy_kwargs=dict(
        lstm_hidden_size=hidden_dim,
        n_lstm_layers=hidden_layers,
        net_arch=dict(pi=[], vf=[]),
        shared_lstm=True,  # Use shared LSTM
        enable_critic_lstm=False,  # Disable separate critic LSTM
    )
)

callback_GAIL = CustomCallback(verbose=0, display_rollout=False, disc_batch_size=128, gail_training=gail_training)

model_post_weight = transfer_weights_from_saved(sw_pretraining, model_pre_weight, True, token_size, hidden_dim, token_size, hidden_layers)

model_post_weight.learn(timesteps, callback=callback_GAIL)
env_GAIL.close()

plot_losses(callback_GAIL.pg_losses, callback_GAIL.value_losses, callback_GAIL.entropy_losses)
plot_sequence_metrics(callback_GAIL.rewards, callback_GAIL.sequence_metrics['wasserstein'],
                      callback_GAIL.sequence_metrics['kl_div'])
plot_discriminator(callback_GAIL.disc_metrics_per_batch['loss'],
                   callback_GAIL.disc_metrics_per_batch['accuracy'],
                   callback_GAIL.disc_metrics_per_batch['accuracy_difference'])