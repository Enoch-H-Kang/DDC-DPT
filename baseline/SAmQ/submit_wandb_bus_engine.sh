# Run All

rm -rfv wandb
wandb sweep configure_bus_engine/sweep_our_state.yaml 2>prophetid.txt #--name 'pqr+our' --project 'state-aggregation'
prophetid=$(cat prophetid.txt | grep Run |sed 's?wandb: Run sweep agent with: wandb agent ??')
# Run prophet
for i in $(seq 14)
do 
	wandb agent $prophetid & 
done
wait


rm -rfv wandb
wandb sweep configure_bus_engine/sweep_pqr.yaml 2>prophetid.txt #--name 'pqr+our' --project 'state-aggregation'
prophetid=$(cat prophetid.txt | grep Run |sed 's?wandb: Run sweep agent with: wandb agent ??')
# Run prophet
for i in $(seq 12)
do 
	wandb agent $prophetid & 
done
wait

rm -rfv wandb
wandb sweep configure_bus_engine/sweep_no_aggregate.yaml 2>prophetid.txt #--name 'pqr+our' --project 'state-aggregation'
prophetid=$(cat prophetid.txt | grep Run |sed 's?wandb: Run sweep agent with: wandb agent ??')
# Run prophet
for i in $(seq 12)
do 
	wandb agent $prophetid & 
done
wait