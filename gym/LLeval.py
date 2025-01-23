import gym
from stable_baselines3 import PPO
#from stable_baselines3 import A2C
#from stable_baselines3 import DQN
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# When loading the model, specify the custom objects
custom_objects = {
    "clip_range": 1,  # This is the default PPO clip range
    "learning_rate": 0.0003  # Default PPO learning rate
}
path = "Expert_policy/LunarLander-v2_PPO.zip"
# Load the model with custom objects
model = PPO.load(path, custom_objects=custom_objects)
#model = A2C.load("LunarLander-v2_A2C.zip")
#model = DQN.load("LunarLander-v2_DQN.zip")
# Create the environment
env = gym.make("LunarLander-v2")

# Evaluation loop without rendering
episodes = 10
for episode in range(episodes):
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")