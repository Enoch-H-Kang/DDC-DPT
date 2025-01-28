import gym
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Configuration Dictionary
CONFIG = {
    "env": "LL",
    "train": True,  # Set to False to skip training
    "total_timesteps": 100_000,  # Number of training steps
    "policy_kwargs": {"net_arch": [64, 64]},  # Neural network architecture
    "verbose": 1,  # Logging level
    "device": "cpu",  # Use "cuda" for GPU
    "beta": 0.95 
}

def create_env(env_id):
    """Creates and returns a Gym environment."""
    return gym.make(env_id)

def train_model(config):
    """Trains the PPO model based on the given configuration."""
    if config["env"] == "LL":
        env_id = "LunarLander-v2"
    elif config["env"] == "AC":
        env_id = "Acrobot-v1"
    elif config["env"] == "CP":
        env_id = "CartPole-v1":
    else:
        raise ValueError(f"Unknown environment: {config['env']}")
    
    env = create_env(env_id)

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
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

        # Save the model
        timestamp = datetime.datetime.now().strftime("%d_%H-%M-%S")
        save_name = f"ppo-lunarlander-{timestamp}"
        model.save(save_name)

    env.close()
    return model

def test_model(model, config):
    """Tests the trained model by running it in the environment."""
    test_env = create_env(config["env_id"])
    obs, info = test_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        done = done or truncated  # Stop if 'truncated' is True

    test_env.close()

def main(config):
    """Main function to train and test the model based on the configuration."""
    model = train_model(config)
    test_model(model, config)

if __name__ == "__main__":
    main(CONFIG)
