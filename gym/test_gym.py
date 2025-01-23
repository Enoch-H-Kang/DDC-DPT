
from gym_train import train  # Make sure this import matches your actual function name

def run_training():
    # Base configuration
    base_config = {
        "env": "zurcher",
        "beta": 0.95,
        "H": 100,
        "seed": 1,
        "shuffle": False,
        "batch_size": 64,
        "lr": 0.001,
        "h_size": 10,
        "n_layer": 2,
        "seed": 1,
        "test": False,
        "store_gpu": True,
        "num_epochs": 1000,
        "layer_norm": False,
        "loss_ratio": 0.5,
        "states_TF": [1,0,0,0,0],
        "num_dummies": 2,
        "dummy_dim": 5
    }

    # Experiments
    experiments = [
        {"num_trajs": 50}
    ]

    for exp in experiments:
        config = {**base_config, **exp}
        print(f"Starting training process for {exp['num_trajs']} trajectories...")
        train(config)
        print(f"training process for {exp['num_trajs']} trajectories completed.")

if __name__ == "__main__":
    run_training()
