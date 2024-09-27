#!/bin/bash

# Get the current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create a new screen session
screen_name="experiment_run_${timestamp}"
screen -dmS $screen_name bash -c "
    # Run the Python script
    python3 run_parallel_pipeline.py --config configs/zurcher_config_240909.json > logs/main_output_${timestamp}.log 2>&1
    
    # After the script finishes, display the contents of the experiment_pids.log file
    echo 'Experiment PIDs and Configurations:' >> logs/main_output_${timestamp}.log
    cat experiment_pids.log >> logs/main_output_${timestamp}.log
    
    # Keep the screen session open
    exec bash
"

echo "Experiments started in screen session: $screen_name"
echo "To attach to this session, use: screen -r $screen_name"
echo "To view the main output log, use: tail -f logs/main_output_${timestamp}.log"
echo "To view a specific experiment log, use: tail -f experiment_X_output.log (replace X with the experiment number)"
echo "To list all screen sessions, use: screen -ls"
echo "To kill the screen session, use: screen -X -S $screen_name quit"


#Run the code with ./run_experiments.sh