import gym
import torch
import datetime
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
from multiHeadedMLPModule import MultiHeadedMLPModule

# Import your custom MLP model
class MLP(MultiHeadedMLPModule):
    def __init__(self,
                 states_dim,
                 actions_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=lambda x: nn.init.constant_(x, 15),
                 layer_normalization=False):
        super().__init__(2, states_dim, [actions_dim, actions_dim], hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization)

    def forward(self, x):
        states = x  # Input states (batch_size, states_dim)
        q_values, vnext_values = super().forward(states)  # Q-values and estimated next-state values

        # Apply softplus to ensure non-negative values
        q_values = F.softplus(q_values)
        vnext_values = F.softplus(vnext_values)
        
        # Compute rewards
        r_values = q_values - 0.95 * vnext_values
        return r_values  # Return the predicted reward tensor


# Configuration
CONFIG = {
    "env": "LL",  # Choose from "LL" (LunarLander), "AC" (Acrobot), "CP" (CartPole)
    "train": True,  # Set to False to skip training
    "total_timesteps": 1000000,  # Number of training steps
    "h_size": 64,  # Hidden layer size
    "n_layer": 2,  # Number of hidden layers
    "layer_norm": False,  # Whether to apply layer normalization
    "policy_kwargs": {"net_arch": [64, 64]},  # Neural network architecture for PPO
    "verbose": 1,  # Logging level
    "device": "cpu",  # Choose device
}

# Select environment and reward model based on CONFIG
if CONFIG['env'] == 'LL':  # LunarLander
    states_dim = 8
    actions_dim = 4
    env_id = "LunarLander-v2"
    model_path = 'models/LL_num_trajs15_lr0.0005_batch128_decay0.001_clip1_20250207.log_rep0_epoch5000.pt'
elif CONFIG['env'] == 'AC':  # Acrobot
    states_dim = 6
    actions_dim = 3
    env_id = "Acrobot-v1"
    model_path = 'models/AC_num_trajs3_lr0.001_batch128_decay0.0001_clip1_20250129.log_rep0_epoch5000.pt'
elif CONFIG['env'] == 'CP':  # CartPole
    states_dim = 4
    actions_dim = 2
    env_id = "CartPole-v1"
    model_path = 'models/CP_num_trajs20_lr0.001_batch128_decay0.001_clip1.1_20250127.log_rep0_epoch2000.pt'
else:
    raise ValueError("Invalid environment! Choose from 'LL', 'AC', or 'CP'.")

# Define model configuration
model_config = {
    'hidden_sizes': [CONFIG['h_size']] * CONFIG['n_layer'],
    'layer_normalization': CONFIG['layer_norm'],
}

# Load trained reward model
device = CONFIG["device"]
reward_model = MLP(states_dim, actions_dim, **model_config).to(device)
reward_model.load_state_dict(torch.load(model_path, map_location=device))
reward_model.eval()  # Set to evaluation mode

# Custom wrapper to replace environment reward with model-predicted reward
class RewardModelWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model

    def step(self, action):
        next_state, _, done, truncated = self.env.step(action)  # Ignore env reward
        state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)  # Use next_state instead of self.env.state

        with torch.no_grad():
            predicted_rewards = self.reward_model(state_tensor.unsqueeze(0))  # Get reward tensor
            predicted_reward = predicted_rewards[0, action].item()  # Select reward for taken action

        return next_state, predicted_reward, done, truncated

def create_env(config):
    """Creates a Gym environment wrapped with the reward model."""
    env = gym.make(env_id)
    return RewardModelWrapper(env, reward_model)

def train_model(config):
    """Trains the PPO model using the custom reward function."""
    env = create_env(config)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=config["policy_kwargs"],
        verbose=config["verbose"],
        device=config["device"],
    )

    if config["train"]:
        model.learn(total_timesteps=config["total_timesteps"])

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

        # Save the trained RL model
        save_name = f"PPO_IRL/PPO-15-{config['env']}"
        model.save(save_name)

    env.close()
    return model

def test_model(model, config):
    """Tests the trained PPO model with the custom reward function."""
    test_env = create_env(config)
    obs = test_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated = test_env.step(action)
        done = done or truncated  # Stop if 'truncated' is True

    test_env.close()

def main(config):
    """Main function to train and test the model based on the configuration."""
    model = train_model(config)
    test_model(model, config)

if __name__ == "__main__":
    main(CONFIG)
