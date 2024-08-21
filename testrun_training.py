import json
from train import train

def run_training():
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
        "lr": 0.001,
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 128,
        "shuffle": True,
        "dropout": 0,
        "test": False,
        "store_gpu": True,
        "num_epochs": 500,
        "batch_size": 64,
        "loss_ratio": 10,
        "infR": True,

    }

    # Experiments
    experiments = [
        {"num_trajs": 500}
    ]

    for exp in experiments:
        config = {**base_config, **exp}
        print(f"Starting training process for {exp['num_trajs']} trajectories...")
        train(config)
        print(f"Training process for {exp['num_trajs']} trajectories completed.")

if __name__ == "__main__":
    run_training()
