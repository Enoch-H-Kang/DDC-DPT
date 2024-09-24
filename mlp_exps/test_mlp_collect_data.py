import json
from mlp_collect_data import generate  # Make sure this import matches your actual function name

def run_mlp_data_collection():
    # Base configuration
    base_config = {
        "env": "zurcher",
        "beta": 0.95,
        "H": 100,
        "maxMileage": 10,
        "numTypes": 1,
        "rollin_type": "expert",
        "theta": [1, 5, 1],
        "seed": 1,
        # Add any MLP-specific parameters here
    }

    # Experiments
    experiments = [
        {"num_trajs": 500},
        {"num_trajs": 5000},
        {"num_trajs": 10000}
    ]

    for exp in experiments:
        config = {**base_config, **exp}
        print(f"Starting MLP data collection process for {exp['num_trajs']} trajectories...")
        generate(config)
        print(f"MLP data collection process for {exp['num_trajs']} trajectories completed.")

if __name__ == "__main__":
    run_mlp_data_collection()