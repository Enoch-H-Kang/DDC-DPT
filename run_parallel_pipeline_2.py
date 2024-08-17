import torch
import argparse
import json
import collect_data_IRL
import train
import ray
import os
from datetime import datetime

@ray.remote(num_gpus=0.25)  # This allows up to 4 experiments per GPU
def run_data_generation_and_training(config, experiment_id):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a CUDA-compatible GPU.")
    
    # Set GPU memory fraction for this process
    torch.cuda.set_per_process_memory_fraction(0.25, device=None)
    
    available_gpu = torch.cuda.current_device()
    pid = os.getpid()

    exp_config_str = ", ".join([f"{k}={v}" for k, v in config.items() if k not in ['global_config', 'training_config', 'zurcher_config']])

    log_filename = f"experiment_{experiment_id}_output.log"

    def log_message(message):
        with open(log_filename, "a") as log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"[{timestamp}] PID {pid} - GPU {available_gpu} - {message}\n")

    # Log the PID and experiment config to a separate file
    with open("experiment_pids.log", "a") as pid_log:
        pid_log.write(f"Experiment {experiment_id}: PID {pid} - Config: {exp_config_str}\n")

    log_message(f"Experiment config ({exp_config_str}): Starting data generation")
    log_message(f"Using GPU: {torch.cuda.get_device_name(available_gpu)}")
    log_message(f"Allocated GPU memory: {torch.cuda.memory_allocated(available_gpu) / 1e9:.2f} GB")

    collect_data_IRL.generate(config)
    
    log_message(f"Experiment config ({exp_config_str}): Completed data generation")
    log_message(f"Experiment config ({exp_config_str}): Starting training")
    
    train.train(config)
    
    log_message(f"Experiment config ({exp_config_str}): Completed training")

    return f"Completed run with experiment config: {exp_config_str}"

def print_kill_instructions():
    print("\nTo kill a specific experiment process:")
    print("1. Check the 'experiment_pids.log' file to find the PID of the experiment you want to terminate.")
    print("2. Use the 'kill' command followed by the PID. For example:")
    print("   kill 3656816")
    print("\nTo kill all experiment processes:")
    print("1. Use the 'pkill' command with the Python interpreter name. For example:")
    print("   pkill python3")
    print("\nNote: Be cautious when using 'pkill' as it will terminate all Python processes, including this script.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)

    global_config = full_config['global_config']
    training_config = full_config['training_config']
    zurcher_config = full_config['zurcher_config']

    if isinstance(zurcher_config['theta'], str):
        zurcher_config['theta'] = json.loads(zurcher_config['theta'])

    experiments = full_config['experiments']

    # Clear the experiment_pids.log file at the start of a new run
    open("experiment_pids.log", "w").close()

    # Initialize Ray with the correct number of GPUs
    num_gpus = torch.cuda.device_count()
    ray.init(num_gpus=num_gpus)

    results = []
    for i, experiment in enumerate(experiments):
        config = {**global_config, **training_config, **zurcher_config, **experiment}
        results.append(run_data_generation_and_training.remote(config, i))

    # Use ray.get() with a timeout to prevent blocking indefinitely
    for result in ray.get(results, timeout=None):
        print(result)

    print("All data generation and training processes are complete.")
    print_kill_instructions()