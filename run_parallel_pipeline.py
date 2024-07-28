#Create a new screen with screen -S mysession
#Detach from the screen with Ctrl+A D
#Reattach to the screen with screen -r mysession
#Kill the screen with exit
#List all screens with screen -ls
#Kill all screens with screen -X quit

#Run this code with 
# python3 run_parallel_pipeline.py --config configs/zurcher_config_240725_1.json > main_output.log 2>&1 &

import torch
import argparse
import json
import collect_data_IRL
import train
import ray

@ray.remote(num_gpus=1)
def run_data_generation_and_training(config):
    # Check for available GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a CUDA-compatible GPU.")
    
    # Get the current available GPU
    available_gpu = torch.cuda.current_device()

    # Create a string of all experiment_config elements
    exp_config_str = ", ".join([f"{k}={v}" for k, v in config.items() if k not in ['global_config', 'training_config', 'zurcher_config']])

    # Log the start of data generation
    with open("main_output.log", "a") as log:
        log.write(f"Experiment config ({exp_config_str}): Starting data generation\n")

    # Data generation
    collect_data_IRL.generate(config)
    
    # Log the completion of data generation and start of training
    with open("main_output.log", "a") as log:
        log.write(f"Experiment config ({exp_config_str}): Completed data generation\n")
        log.write(f"Experiment config ({exp_config_str}): Starting training\n")
    
    # Training
    train.train(config)
    
    # Log the completion of training
    with open("main_output.log", "a") as log:
        log.write(f"Experiment config ({exp_config_str}): Completed training\n")

    return f"Completed run with experiment config: {exp_config_str}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)

    global_config = full_config['global_config']
    training_config = full_config['training_config']
    zurcher_config = full_config['zurcher_config']

    # Ensure theta is a list of numbers
    if isinstance(zurcher_config['theta'], str):
        zurcher_config['theta'] = json.loads(zurcher_config['theta'])

    experiments = full_config['experiments']

    # Initialize Ray
    ray.init()

    # Submit tasks to Ray
    results = []
    for experiment in experiments:
        # Merge configurations for each experiment
        config = {**global_config, **training_config, **zurcher_config, **experiment}
        results.append(run_data_generation_and_training.remote(config))

    # Wait for all tasks to complete
    ray.get(results)

    print("All data generation and training processes are complete.")