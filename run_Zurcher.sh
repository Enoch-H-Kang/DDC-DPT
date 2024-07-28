# Collect data

#python3 collect_data.py --env Zurcher --bustotal 10 --beta 0.95 --theta [1,2,9] --H 10 --maxMileage 10 --numTypes 4 --extrapolation False
python3 collect_data_IRL.py --env Zurcher --bustotal 10 --beta 0.95 --theta [1,2,9] --H 10 --maxMileage 10 --numTypes 4 --extrapolation False
# Train
python3 train.py --env Zurcher --bustotal 10 --beta 0.95 --theta [1,2,9] --H 10 --numType 4 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1

# Evaluate, choose an appropriate epoch
python3 eval.py --env Zurcher --bustotal 100 --beta 0.95 --theta [1,2,9] --H 100 --numType 4  --lr 0.001 --layer 4 --head 4 --shuffle --n_eval 200 --epoch 200 --seed 1



#Run the code in background and store the output in a log file train_output.log

nohup python3 collect_data_IRL.py --env Zurcher --bustotal 1000 --beta 0.95 --theta "[1,5,1]" --H 100 --maxMileage 10 --numTypes 1 --extrapolation False > collect_data_IRL_output.log 2>&1 &
nohup python3 train.py --env Zurcher --bustotal 1000 --beta 0.95 --theta "[1,5,1]" --H 100 --numType 1 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1 > train_output.log 2>&1 &

#Disown the process so that it doesn't get killed when the terminal is closed
disown % 1767977

#When you want to kill the code in background
#Step 1: check the list of jobs with their respective job numbers and process IDs (PIDs).
jobs -l
#Step 2: kill the task
kill 1767977
#Force terminate if necessary:
kill -9 1767977

#Check all the processes running behind
ps aux | grep train.py



############### Screen + parallel processing #############

#Create a new screen with screen -S mysession
#Detach from the screen with Ctrl+A D
#Reattach to the screen with screen -r mysession
#Kill the screen with exit
#List all screens with screen -ls
#Kill all screens with screen -X quit

#Run this code with 
python3 run_parallel_pipeline.py > main_output.log 2>&1 &