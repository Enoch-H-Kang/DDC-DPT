import gym
import numpy as np
import pickle
import os
from stable_baselines3 import PPO
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_lunarlander_histories(num_trajs, model):
    """
    Generate LunarLander trajectories for a given number of episodes.
    """
    env = gym.make("LunarLander-v2")
    trajs = []
    
    with tqdm(total=num_trajs, desc=f"Generating {num_trajs} trajectories") as pbar:
        for _ in range(num_trajs):
            # Reset environment
            state = env.reset()
            
            # Initialize buffers for current episode
            states = []
            actions = []
            next_states = []
            rewards = []
            
            done = False
            while not done:
                action, _ = model.predict(state, deterministic=True)
                next_state, reward, done, _ = env.step(action)
                
                states.append(state.copy()) 
                actions.append(action.copy())
                next_states.append(next_state.copy())
                rewards.append(reward) 
                
                state = next_state
            
            # Save completed trajectory
            traj = {
                'states': np.array(states),
                'actions': np.array(actions),
                'next_states': np.array(next_states),
                'rewards': np.array(rewards)
            }
            trajs.append(traj)
            pbar.update(1)
    
    env.close()
    return trajs

def build_data_filename(config, mode):
    """
    Builds the filename for the dataset.
    Mode is either 'train', 'test', or 'eval'.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}")
    
    filename += f'_{mode}'
    
    return filename_template.format(filename)

if __name__ == "__main__":
    # Create datasets directory if it does not exist
    os.makedirs("datasets", exist_ok=True)
    
    # Configuration dictionary
    config = {
        "env": "LunarLander-v2",
        "num_trajs": 5000,  # Total trajectories
    }
    
    # Split train/test trajectories
    NUM_TRAIN_TRAJECTORIES = int(config["num_trajs"] * 0.8)
    NUM_TEST_TRAJECTORIES = config["num_trajs"] - NUM_TRAIN_TRAJECTORIES

    path = "Expert_policy/LunarLander-v2_PPO.zip"
    # Load the trained PPO model
    try:
        custom_objects = {"clip_range": 1, "learning_rate": 0.0003}
        model = PPO.load(path, custom_objects=custom_objects)
    except FileNotFoundError:
        print("Error: Could not find the trained PPO model file")
        exit(1)
    
    # Generate train/test trajectories
    print(f"Generating {NUM_TRAIN_TRAJECTORIES} training trajectories...")
    train_trajs = generate_lunarlander_histories(NUM_TRAIN_TRAJECTORIES, model)

    print(f"Generating {NUM_TEST_TRAJECTORIES} testing trajectories...")
    test_trajs = generate_lunarlander_histories(NUM_TEST_TRAJECTORIES, model)

    # Generate filenames using `config`
    train_filepath = build_data_filename({**config}, 'train')
    test_filepath = build_data_filename({**config}, 'test')

    # Save the trajectories
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)

    print(f"Saved training data to {train_filepath}.")
    print(f"Saved testing data to {test_filepath}.")
