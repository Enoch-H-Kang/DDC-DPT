
from gym_train import train  # Make sure this import matches your actual function name

def run_training():
    # Base configuration
    base_config = {
        "env": "LL",
        "beta": 0.95,
        "H": 100,
        "seed": 1,
        "shuffle": False,
        "batch_size": 32,
        "lr": 0.001,
        "h_size": 256,
        "n_layer": 2,
        "decay": 0.0001,
        "clip":1,
        "Tik": True,
        "seed": 1,
        "test": False,
        "store_gpu": True,
        "num_epochs": 1000,
        "layer_norm": False,
        "repetitions": 5,
        
    }

    # Experiments
    experiments = [
        {"num_trajs": 1000}
    ]

    for exp in experiments:
        config = {**base_config, **exp}
        print(f"Starting training process for {exp['num_trajs']} trajectories...")
        train(config)
        print(f"training process for {exp['num_trajs']} trajectories completed.")

if __name__ == "__main__":
    run_training()
