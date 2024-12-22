
from airline_train import train  # Make sure this import matches your actual function name

def run_training():
    # Base configuration
    base_config = {
        "env": "airline",
        "beta": 0.95,
        "seed": 1,
        "shuffle": False,
        "batch_size": 32,
        "lr": 0.001,
        "h_size": 10,
        "n_layer": 2,
        "seed": 1,
        "test": False,
        "store_gpu": True,
        "num_epochs": 1000,
        "layer_norm": False,
        "loss_ratio": 0.5,
        "repetitions": 1,
        "proj": True,
        "div": 4,
    }

    # Experiments
    experiments = [
        {"carr_id": 1}
    ]

    for exp in experiments:
        config = {**base_config, **exp}
        print(f"Starting training process for Carrier {exp['carr_id']}...")
        train(config)
        print(f"training process for Carrier {exp['carr_id']} completed.")

if __name__ == "__main__":
    run_training()
