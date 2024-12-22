import json
from collect_data_IRL import generate

def run_data_collection():
    # Base configuration
    base_config = {
        "env": "zurcher",
        "beta": 0.95,
        "H": 100,
        "maxMileage": 10,
        "numTypes": 1,
        "rollin_type": "expert",
        "theta": [1, 5, 1],
        "seed": 1
    }

    # Experiments
    experiments = [
        {"num_trajs": 1000},
        {"num_trajs": 2000}
    ]

    for exp in experiments:
        config = {**base_config, **exp}
        print(f"Starting data collection process for {exp['num_trajs']} trajectories...")
        generate(config)
        print(f"Data collection process for {exp['num_trajs']} trajectories completed.")

if __name__ == "__main__":
    run_data_collection()