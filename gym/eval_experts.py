import gym
import os
from stable_baselines3 import PPO
#from stable_baselines3 import A2C
#from stable_baselines3 import DQN
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# When loading the model, specify the custom objects
custom_objects = {
    "clip_range": 1,  # This is the default PPO clip range
    "learning_rate": 0.0003  # Default PPO learning rate
}

env_name = 'LL'


if env_name == 'LL':
    path = "Expert_policy/LunarLander-v2_PPO.zip"
    env = gym.make("LunarLander-v2")
    state_dim = 8
    action_dim = 4
    save_path = "csvs/LL_experts.csv"
elif env_name == 'CP':
    path = "Expert_policy/CartPole-v1_PPO.zip"
    env = gym.make("CartPole-v1")
    state_dim = 4
    action_dim = 2
    save_path = "csvs/cart_pole_data.csv"
elif env_name == 'AC':
    path = "Expert_policy/Acrobot-v1_PPO.zip"
    env = gym.make("Acrobot-v1")
    state_dim = 6
    action_dim = 3
    save_path = "csvs/acrobot_data.csv"
else:
    print('Invalid environment')
    exit()
    
# Load the model with custom objects
model = PPO.load(path, custom_objects=custom_objects)
# Create the environment


data = []
results = []
# Evaluation loop without rendering


episodes = 1000


for episode in range(episodes):
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs)
        if env_name == 'LL':   
            if (obs[6] ==1) & (obs[7] ==1):
                action = 0
        next_obs, reward, done, info = env.step(action)
        
        '''
        if env_name == 'LL':   
            if (obs[6] ==1) & (obs[7] ==1):
                total_reward += 100
                print('landed')
                done = True
        '''
        data.append([episode]+ list(obs) + [action] + [reward] + list(next_obs))
        obs = next_obs
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    results.append({'Episode': episode + 1, 'Total_reward': total_reward})

columns = ["Episode"] + [f"s{i}" for i in range(state_dim)] + ["action", "reward"] + [f"s'_{i}" for i in range(state_dim)]
reward_columns = ["Episode", "Total_reward"]
# Save to CSV
#df = pd.DataFrame(data, columns=columns)
#df = df.round(3)
df = pd.DataFrame(results, columns=reward_columns)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

df.to_csv(save_path, index=False)