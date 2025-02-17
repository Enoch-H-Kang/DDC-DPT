# ERM-IRL

## Overview
This repository contains code for running **Zurcher** and **Gym-based** environments, generating data, and training reinforcement learning models. It supports:
- **Zurcher bus engine replacement simulation (bus folder):** A standard benchmark in Econometrics literature.
- **Gym Environments (gym folder):** Training and evaluating policies using **Stable-Baselines3**.
---

### **3. Install Dependencies**
For each of Zurcher and Gym, the requirements.txt is different.
```bash
pip install -r requirements.txt
```

## **Code Structure**
```
ERM-IRL/
│── bus/                      # Zurcher (Bus) Environment
│   ├── Zurcher_train.py      # Training script
│   ├── Zurcher_collect_data.py # Data collection script
│   ├── run_Zurcher_ray.py    # Main execution script
│   ├── mlp.py                # MLP model implementation
│   ├── multiHeadedMLPModule.py # Multi-headed MLP module
│   ├── requirements.txt      # Dependencies
│   ├── parametric/           # Parametric testing
│   │   ├── gen_data.py       # Generate data for parametric testing
│   │   ├── estimation.py     # Estimation of CCP (ML-IRL, forward method, NFXP)
│   ├── datasets/             # Pre-generated dataset to save the time for simulation   
│   ├── configs/              # Configuration files for bus environment
│── gym/                      # Gym Environment
│   ├── gym_gen.py            # Data generation for Gym environment
│   ├── gym_train.py          # Training script for Gym-based models
│   ├── gym_eval.py           # Evaluation script
│   ├── mlp.py                # MLP model implementation
│   ├── multiHeadedMLPModule.py # Multi-headed MLP module
│   ├── run_gym_ray.py        # Main execution script
│   ├── Expert_policy/        # PPO-trained expert policies (Stable-Baselines3)
│   ├── configs/              # Configuration files for gym environment
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

---

## **Usage Instructions**

### **1. Running Zurcher data generation & Training**

```bash
python3 bus/run_Zurcher_ray.py --config configs/config1.json
```

### **2. Running Gym Environment Training & Evaluation**

#### **(a) Run the Complete Gym data generation & Training**
```bash
python3 gym/run_gym_ray.py --config gym/configs/gym.json
```

#### **(b) Evaluate the Gym Policy**
```bash
python3 gym/gym_eval.py 
```

---

## **Configuration Files**
- **`bus/configs/config1.json`** → Used for Zurcher execution.
- **`gym/configs/gym.json`** → Used for Gym environment execution.

Example **Zurcher configuration file (`config1.json`)**:
```json
{
  "global_config": {
    "env": "zurcher",
    "beta": 0.95,
    "H": 100
  }
  ,
  "training_config": {
  "h_size": 50,
  "n_layer": 2,
  "shuffle": false,
  "seed": 1,
  "test": false,
  "store_gpu": true,
  "layer_norm": false,
  "num_epochs": 20000,
  "repetitions": 1,
  "lr":0.001,
  "decay": 0.0005,
  "clip":1.00,
  "Tik": false
  }
  ,
  "zurcher_config": {
    "maxMileage": 10,
    "rollin_type": "expert",
    "theta": [1, 5, 1],
    "numTypes": 1,
    "dummy_dim": 10,   
    "num_trajs": 1000,
    "batch_size": 32
  },
  "experiments": [
    {
      "num_dummies": 2
    }
  ]
}
```
Ignore numTypes.
dummy_dim: If 10, then -10, -9, ... 10 is the number of possible dummy values for each dummy dimension.
num_dummies: If 2, then we will have 2 dummy dimensions.
maxMileage: The maximum mileage the bus can reach. After this point, mileage of a bus does not increase.
theta: [theta0, theta1, theta2], where theta0 is per-mileage maintenance cost and theta1 is replacement cost. Ignore theta2. 

Example **Gym configuration file (`gym.json`)**:
```json
{
  "global_config": {
  "beta": 1
  }
  ,
  "training_config": {
  "h_size": 64,
  "n_layer": 2,
  "shuffle": true,
  "seed": 1,
  "test": false,
  "store_gpu": true,
  "layer_norm": false,
  "num_epochs": 10000,
  "repetitions": 1,
  "lr":0.0005,
  "Tik": false,
  "decay": 0.001,
  "clip": 1
  }
  ,
  "env_config": {
    "env": "CP"
  },
  "experiments": [
    {
       "num_trajs": 15, "batch_size": 256
    }
  ]
}
```
For Cartpole, use "env": "CP"
For Lunar Lander, use "env": "LL"
For Acrobot, use "env": "AC"
---

## **Pretrained Policies**
- The **Expert_policy/** directory contains **PPO-trained policies** from **Stable-Baselines3** for Gym environments.
---

