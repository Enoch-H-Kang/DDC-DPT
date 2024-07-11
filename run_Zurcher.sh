# Collect data
python3 collect_data.py --env Zurcher --Bustotal 100 --beta 0.95 --theta [1,2,9] --H 100 --maxMileage 200 --numTypes 4 --extrapolation False

# Train
python3 train.py --env Zurcher --Bustotal 100 --beta 0.95 --theta [1,2,9] --H 100 --numType 4 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1

# Evaluate, choose an appropriate epoch
python3 eval.py --env Zurcher --Bustotal 100 --beta 0.95 --theta [1,2,9] --H 100 --numType 4  --lr 0.001 --layer 4 --head 4 --shuffle --epoch 200 --seed 1