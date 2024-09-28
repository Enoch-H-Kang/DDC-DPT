import glob
import os
from pathlib import Path

import numpy as np
import torch
from SAmQ.helper.util import Log
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from SAmQ.helper.rewards import linear_reward
from SAmQ.env.bus_env import syn_bus_env
from SAmQ.env.airline_env import airline_env
from SAmQ.rl.softQ import soft_q_iteration
from SAmQ.irl.pqr import pqr_aggregation
from SAmQ.rl.ddm import rust_ddm
from scipy import optimize
import wandb
from SAmQ.dataClass.irl_class import pqr_dataclass, airline_dataclass


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # pick reward
    if args.reward_type == 'linear':
        r_f = linear_reward

    # Choose environment
    if args.env_name == 'bus_engine':
        args.d_s = len(args.theta)

        env = syn_bus_env(n_a=args.n_a, d_s=args.d_s, b=args.b, theta=args.theta, r_f=r_f, discount=args.discount)
        log_path = Path(args.log_dir) / args.env_name / (args.method+'delta'+str(args.delta))
        log = Log(log_path, vars(args))
        log(f'Log dir: {log.dir}')
        writer = SummaryWriter(log.dir)
        log('Generate Data - Soft Q')
        softq = soft_q_iteration(discount=args.discount, optimizer = torch.optim.Adam, l_r=args.learning_rate,
            hidden_size=args.hidden_size, env=env, log = log)
        softq_dict = softq.train(n_epochs=args.n_steps*2, batch_size=args.batch_size, epoch_size=args.epoch_size,
            num_workers = 0, patience = args.patience, writer=writer)
        re_dict = softq_dict

        log('Generate Data - Saving')
        n_states = softq.generate_data(args.n_sample)
        re_dict['Number of states'] = n_states
        data_class = pqr_dataclass
    elif args.env_name=='airline':
        log_path = Path(args.log_dir) / args.env_name / (args.method+'delta'+str(args.delta))
        log = Log(log_path, vars(args))
        writer = SummaryWriter(log.dir)

        data_path = Path('ailrline_data')
        env = airline_env(data_path, args.carr_id, r_f, log.dir)

        re_dict = {'Number of states':env.n_states, 'd_s':env.d_s}
        data_class = airline_dataclass

        args.d_s = env.d_s
        args.theta = [1]*args.d_s
        args.theta_0 = [1]*args.d_s
    else:
        raise NotImplementedError

    pqr_mod = pqr_aggregation(data = data_class(log.dir / 'data_generation'), discount=args.discount, optimizer=torch.optim.Adam,
        l_r=args.learning_rate, hidden_size=args.hidden_size, env=env, log=log)

    if args.method == 'pqr' or args.method == 'our':
        log('Run PQR')
        p_dict = pqr_mod.train_q(nEpochs=args.n_steps, batch_size=args.batch_size, num_workers=0, patience=args.patience, writer=writer)
        re_dict = {**re_dict, **p_dict}
        print(re_dict)
        q_dict = pqr_mod.train_q_star(n_steps=args.n_steps*2, batch_size=args.batch_size,writer=writer,num_workers = 0, patience=args.patience)
        re_dict = {**re_dict,**q_dict}
    if args.method=='pqr':
        re_dict['test_likelihood'] = pqr_mod.get_test_likelihood()
        r_dict = pqr_mod.train_r(n_steps=args.n_steps, batch_size=args.batch_size,writer=writer,num_workers = 0, patience=args.patience)
        re_dict = {**re_dict,**r_dict}
        reward_mse = ((np.array(pqr_mod.linear_fit()) - np.array(args.theta))**2).mean()
    
    #SAmQ as a base method
    elif args.method == 'our' or args.method == 'state' or args.method == 'no_aggregate':
        log('Run Aggregation')
        
        
        if args.method == 'no_aggregate':
            args.n_states_aggregated = re_dict['Number of states']
        n_states_aggregated = pqr_mod.aggregate(args.delta, args.method,args.n_states_aggregated)
        re_dict['number of states after aggregation'] = n_states_aggregated
        data_class(log.dir / 'data_generation',str(args.delta)+args.method)
        
        
        #base_method
        if args.base_method == 'pqr':
            pqr_mod = pqr_aggregation(data = data_class(log.dir / 'data_generation',str(args.delta)+args.method), discount=args.discount, optimizer=torch.optim.Adam,
                l_r=args.learning_rate, hidden_size=args.hidden_size, env=env, log=log, aggregate_or_not=True)
            log('Run PQR')
            p_dict = pqr_mod.train_q(nEpochs=args.n_steps, batch_size=args.batch_size, num_workers=0, patience=args.patience, writer=writer)
            re_dict = {**re_dict, **p_dict}
            print(re_dict)
            q_dict = pqr_mod.train_q_star(n_steps=args.n_steps*2, batch_size=args.batch_size,writer=writer,num_workers = 0, patience=args.patience)
            re_dict = {**re_dict,**q_dict}
            re_dict['test_likelihood'] = pqr_mod.get_test_likelihood()
            r_dict = pqr_mod.train_r(n_steps=args.n_steps, batch_size=args.batch_size,writer=writer,num_workers = 0, patience=args.patience)
            re_dict = {**re_dict,**r_dict}
            reward_mse = ((np.array(pqr_mod.linear_fit()) - np.array(args.theta))**2).mean()
        elif args.base_method == 'mle':
            log('Run Rust DDM')
            def f(theta):
                ddm_mod = rust_ddm(discount=args.discount, log=log, env=env, delta=args.delta, r_f =r_f,
                    aggregation_method=args.method)
                ddm_mod.get_r(args.b, theta)
                estiamted_dict = ddm_mod.get_q(n_steps=args.n_steps, alpha=0.1)
                train_likelihood = ddm_mod.get_p()
                test_likelihood = ddm_mod.get_p_test()
                log(f'Input theta is {np.array(theta)}')
                log(f'Objective value is {-train_likelihood}')
                log(f'test_likelihood  is {test_likelihood}')
                re_dict['test_likelihood'] = test_likelihood
                re_dict['train_likelihood'] = train_likelihood
                return -train_likelihood
            f(args.theta)
            f(args.theta_0)
            solver = optimize.minimize(f, args.theta_0)
            reward_mse = ((np.array(solver.x) - np.array(args.theta))**2).mean()
        else:
            raise NotImplementedError
        
    else:
        raise NotImplementedError
    log.close()
    writer.close()
    reward_dict = {'reward_mse':reward_mse}
    re_dict = {**re_dict, **reward_dict}
    print(re_dict)

    return re_dict


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--n-steps', type=int, default=100)
    parser.add_argument('--env-name', default='bus_engine')
    parser.add_argument('--log-dir', default='data')
    parser.add_argument('--reward-type', default='linear')
    parser.add_argument('--method', type =str, default = 'our')
    parser.add_argument('--base-method', type =str, default = 'mle')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--n-a', type=int, default=2)
    parser.add_argument('--b', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=10)
    parser.add_argument('--n-sample', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epoch-size', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=0.004)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--n-states-aggregated', type=int, default=5)
    parser.add_argument('--theta', nargs='+', type=float, default = [1,0,0]) #Non-dummy or not.
    parser.add_argument('--theta-0', nargs='+', type=float, default = [0.5,0.5,0.5])
    parser.add_argument('--carr-id', type = int, default = 1)




    wandb.init(config = parser.parse_args())
    re_dict = run(parser.parse_args())
    wandb.log(re_dict)
