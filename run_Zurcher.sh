# Collect data
#python3 collect_data.py --env Zurcher --bustotal 10 --beta 0.95 --theta [1,2,9] --H 10 --maxMileage 10 --numTypes 4 --extrapolation False
python3 collect_data_IRL.py --env Zurcher --bustotal 10 --beta 0.95 --theta [1,2,9] --H 10 --maxMileage 10 --numTypes 4 --extrapolation False
# Train
python3 train.py --env Zurcher --bustotal 10 --beta 0.95 --theta [1,2,9] --H 10 --numType 4 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1

# Evaluate, choose an appropriate epoch
python3 eval.py --env Zurcher --bustotal 100 --beta 0.95 --theta [1,2,9] --H 100 --numType 4  --lr 0.001 --layer 4 --head 4 --shuffle --n_eval 200 --epoch 200 --seed 1