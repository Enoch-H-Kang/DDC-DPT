#Create a new screen with screen -S mysession
#Detach from the screen with Ctrl+A D
#Reattach to the screen with screen -r mysession
#Kill the screen with exit
#List all screens with screen -ls
#Kill all screens with screen -X quit

#Run this code with 
# python3 run_parallel_pipeline.py > main_output.log 2>&1 &


import multiprocessing
import subprocess
import time
import torch
import argparse
import json
import collect_data_IRL

def run_data_generation_and_training(config):
    # Check for available GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a CUDA-compatible GPU.")
    
    # Get the current available GPU
    available_gpu = torch.cuda.current_device()


    # Data generation command
    #data_gen_command = f"CUDA_VISIBLE_DEVICES={available_gpu} python3 collect_data_IRL.py --env Zurcher --bustotal {bustotal} --beta 0.95 --theta \"[1,5,1]\" --H 100 --maxMileage 10 --numTypes 1 --extrapolation False"
    collect_data_IRL(config)
    
    # Training command
    #log_file = f"train_output_{bustotal}.log"
    train_command = f"CUDA_VISIBLE_DEVICES={available_gpu} python3 train.py --env Zurcher --bustotal {bustotal} --beta 0.95 --theta \"[1,5,1]\" --H 100 --numType 1 --lr 0.001 --layer 12 --head 8 --shuffle --seed 1 > {log_file} 2>&1"

    # Log the start of data generation
    with open("main_output.log", "a") as log:
        log.write(f"Bustotal {bustotal}: Starting data generation\n")
    
    # Run data generation
    subprocess.run(data_gen_command, shell=True, check=True)
    
    # Log the completion of data generation
    with open("main_output.log", "a") as log:
        log.write(f"Bustotal {bustotal}: Completed data generation\n")
        log.write(f"Bustotal {bustotal}: Starting training\n")
    
    # Run training
    subprocess.run(train_command, shell=True, check=True)
    
    # Log the completion of training
    with open("main_output.log", "a") as log:
        log.write(f"Bustotal {bustotal}: Completed training\n")

if __name__ == "__main__":
    #bustotals = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000]
    # bustotals = [1000, 2000, 3000]
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)

    with open(args.config, "r") as f:
        config = json.load(f)

    # Create a pool with a maximum of 2 worker processes
    with multiprocessing.Pool(processes=2) as pool:
        # Map the function to the list of bustotals
        pool.map(run_data_generation_and_training, config)

    print("All data generation and training processes are complete.")
