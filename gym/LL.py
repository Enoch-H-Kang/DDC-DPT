#import gymnasium as gym
import gym
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


env_id = "LunarLander-v2"

# If you want to render the environment, set render_mode="human"
# Note: remove render_mode if you're training on a headless machine or server
env = gym.make(env_id)

timestamp = datetime.datetime.now().strftime("%d_%H-%M-%S")

# Create the model
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# Train the model
model.learn(total_timesteps=100_000)

# Evaluate the model (optional)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Save model locally (optional)
save_name = f"sac-lunarlander-{timestamp}"
model.save(save_name)

# Test run with rendering
test_env = gym.make(env_id)
obs, info = test_env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = test_env.step(action)
    done = done or truncated  # If 'truncated' is True, also end the episode

test_env.close()

##############################################################################
# (Optional) Upload the trained model to Hugging Face Hub
##############################################################################
# 1) Hugging Face CLI login
# huggingface-cli login
#
# 2) Modify this code if you want to push the model:

