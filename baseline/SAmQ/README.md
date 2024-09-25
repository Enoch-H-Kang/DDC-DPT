# State Aggregation Minimizing Q error

State aggregation minimizing Q error (SAmQ) is a toolkit to conduct state aggregation for dynamic discrete choice model estimation. 

## Installation

```
pip install git+https://github.com/gengsinong/SAmQ.git@master
```

## Main Funcion

The main function is in [`main.py`](https://github.com/gengsinong/SAmQ/blob/master/main.py). It includes a list of parameters.
```
python main.py
```

## Tuning

The tunning is conducted by [`submit_wandb_airline.sh`](https://github.com/gengsinong/SAmQ/blob/master/submit_wandb_airline.sh), 
[`submit_wandb_bus_engine.sh`](https://github.com/gengsinong/SAmQ/blob/master/submit_wandb_bus_engine.sh) 
and [`submit_wandb_bus_engine_plot.sh`](https://github.com/gengsinong/SAmQ/blob/master/submit_wandb_bus_engine_plot.sh). 
This file uses [wandb](https://wandb.ai/site). 


## Data Generation

### Bus Engine Replacement Analysis
To generate the decision-making data for dynamic discrete choice model estimation, we consider a synthetic setting similar to the bus engine application in the [Rust model](https://www.google.com/search?q=rust+bus+engine&rlz=1C1CHBF_zh-TWUS812US812&oq=rust+bus+engine&aqs=chrome..69i57j0i22i30.2014j0j7&sourceid=chrome&ie=UTF-8). 
The specific setting is defined and detialed in [`env.bus_env`](https://github.com/gengsinong/SAmQ/blob/master/SAmQ/env/bus_env.py). 
Then, we conduct soft-q iteration to estimate the optimal choice-specific value function, and generate data in [`rl.softQ`](https://github.com/gengsinong/SAmQ/blob/master/SAmQ/rl/softQ.py). 

### Airline Market Entry analysis
Experiments on airline market entry analysis leverages the data in [`airline_data`](https://github.com/gengsinong/SAmQ/tree/master/ailrline_data).

## State Aggregation
The state aggregation step is conducted in [`irl.pqr`](https://github.com/gengsinong/SAmQ/blob/master/SAmQ/irl/pqr.py). 
The aggregation uses the [deep PQR method](https://arxiv.org/abs/2007.07443).
