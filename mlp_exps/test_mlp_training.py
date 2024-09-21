
from mlp_train import train  # Make sure this import matches your actual function name

def run_mlp_training():
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
        "shuffle": False,
        "batch_size": 64,
        "lr": 0.001,
        "h_size": 10,
        "n_layer": 2,
        "seed": 1,
        "test": False,
        "store_gpu": True,
        "num_epochs": 20000,
        "layer_norm": False,
        "loss_ratio": 1.111          
    }

    # Experiments
    experiments = [
        {"num_trajs": 10000}
    ]

    for exp in experiments:
        config = {**base_config, **exp}
        print(f"Starting MLP training process for {exp['num_trajs']} trajectories...")
        train(config)
        print(f"MLP training process for {exp['num_trajs']} trajectories completed.")

if __name__ == "__main__":
    run_mlp_training()
