"""
Main playground module for pq and r and squash
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn
from torch.nn import functional as F  # NOQA
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

import datetime
from SAmQ.dataClass.irl_class import pqr_dataclass
from torch.utils.data import Dataset, DataLoader
import pickle
from SAmQ.helper.functions import _Default, make_optimizer, makePath,setupTensorboard, dumpAllTensorboard
from SAmQ.estimation.qEstimation import qEstimation

from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import cKDTree as KDTree
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from dowel import logger, tabular

from tqdm import tqdm
from sklearn.neighbors import NearestCentroid


class pqr_aggregation():
    def __init__(self, data, env, log, hidden_size, optimizer, l_r, discount, aggregate_or_not=False):
        self.n_a = env.n_a
        if aggregate_or_not:
            self.d_s = 1
        else:
            self.d_s = env.d_s
        self.log = log
        self.root_path = log.dir
        self.data = data
        self.q = qEstimation(self.d_s,self.n_a,hidden_sizes=[hidden_size,hidden_size], hidden_nonlinearity=F.relu)
        self.q_star = qEstimation(self.d_s,1,hidden_sizes=[hidden_size,hidden_size],hidden_nonlinearity=F.relu)
        self.r = qEstimation(self.d_s,self.n_a, hidden_sizes=[hidden_size,hidden_size],hidden_nonlinearity=F.relu)
        self.path = self.log.dir / 'train'
        self.aggregate_or_not = aggregate_or_not
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # Setting up the training parameters for deep models. 
        self.q_optimizer = make_optimizer(optimizer, self.q, lr=l_r, amsgrad = True)
        self.q_star_optimizer = make_optimizer(optimizer, self.q_star, lr=l_r, amsgrad = True)
        self.r_optimizer = make_optimizer(optimizer, self.r, lr=l_r, amsgrad = True)
        self.discount = discount

    def train_q(self, nEpochs, batch_size, writer, num_workers=0, patience = 5):
        # Train policy estiamtor. 
        # Policy estiamtor has an energy form. 
        # The deep function is what inside the exponential. 

        dataloaders = DataLoader(self.data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Specify loss function
        lossF = CrossEntropyLoss()    # MLE
        min_loss = np.inf
        waiting = 0
        for epoch in tqdm(range(nEpochs)):
            epoch_loss = 0
            q_mse = 0
            p_loss = 0
            count = 0
            for s,a,p,_,_,q,_,_ in dataloaders: 
                #est = self.q.forward(s)
                est = torch.clamp(self.q.forward(s), min = -100, max=100)
                self.q_optimizer.zero_grad()    
                lossV = lossF(est,a[:,0])

                lossV.backward()    # Get the gradient from the loss
                self.q_optimizer.step()      # Use the optimizer and the gradient to update the deep.
                epoch_loss = epoch_loss + lossV
                count+=1
                
                # other metrics
                exp_est = torch.exp(est)
                p_est=exp_est/exp_est.sum(1).reshape(len(est),1)
                p_loss +=((p-p_est)**2).mean() 
            train_q_dict = {'Mean Training Loss for Q estimation':epoch_loss.item()/count, 
                'Probability Loss (unknown in practice)':p_loss.item()/count}
            for k, value in train_q_dict.items():
                writer.add_scalar('train_q/'+k, value, epoch)
            notStop = epoch_loss < min_loss
            if notStop:
                min_loss =epoch_loss
                waiting = 0
                torch.save(self.q.state_dict(), self.path / 'q_best_model.pth')
            else:
                waiting+=1
            if waiting >patience:
                break
        self.q.load_state_dict(torch.load(self.path / 'q_best_model.pth'))
        # get train lileihood
        q_est = torch.clamp(self.q(torch.FloatTensor(self.data.s)), min = -100, max=100).detach().numpy()
        #q_est = self.q(torch.FloatTensor(self.data.s).detach()).numpy()
        a = self.data.a
        train_q_dict['train_likelihood'] = self.get_likelihood(q_est, a)
        return train_q_dict

    def train_q_star(self, n_steps, batch_size,writer,num_workers = 0, patience = 5):
        # Trying to get the Q for non Prime (y==0)
        # Algorithm 2 in the PQR.pdf
        # According to the doc, Q^* solves a fixed point. 

        dataloaders = DataLoader(self.data, batch_size=batch_size,shuffle=True, num_workers=num_workers)
        min_loss = np.inf
        waiting = 0
        lossF = torch.nn.MSELoss()
        for epoch in tqdm(range(n_steps)):
            epoch_loss = 0
            count = 0
            mse_q_star = 0
            mse_q = 0
            #trueMse = 0
            for s, a, p,p_next,s_next, q, q_next,_ in dataloaders:
                count+=1
                idx = (a==0)[:,0]    # get anchor actions
                s = s[idx,:]
                s_next = s_next[idx,:]
                q = q[idx,:]
                q_next = q_next[idx,:]
                p = p[idx,:]
                p_next = p_next[idx,:]
                q_star_next_est = self.q_star(s_next)
                q_next_est = self.q(s_next)
                p_next_est = F.softmax(q_next_est, dim=1)

                y = self.discount*(-torch.log(p_next_est[:,0])+q_star_next_est[:,0])[:,None]   #line 5 of algorithm 2
                #y = self.discount*(-torch.log(lN[:,0])+qNext[:,0])[:,None]   #line 5 of algorithm 2
                est = self.q_star.forward(s)
                lossV = lossF(est,y)
                self.q_star_optimizer.zero_grad()
                lossV.backward()
                self.q_star_optimizer.step()
                epoch_loss = epoch_loss + lossV
                
                q_est = self.q(s)
                q_est_anchored = (q_est - q_est[:,0][:,None])+est[:,None]
                mse_q_star+= ((est - q[:,0])**2).mean()
                mse_q+= ((q_est_anchored - q)**2).mean()
            train_q_star_dict = {'Mean Training Loss for q_star':epoch_loss.item()/count,
                'MSE for q_star (unknown in practice)':mse_q_star.item()/count,
                'MSE for q (unknown in practice)':mse_q/count}
            for k, value in train_q_star_dict.items():
                writer.add_scalar('train_q/'+k, value, epoch)
            notStop = epoch_loss < min_loss
            if notStop:
                min_loss =epoch_loss
                waiting = 0
                torch.save(self.q_star.state_dict(), self.path/'q_star_best_model.pth') 
            else:
                waiting+=1
            if waiting >patience:
                break
        self.q_star.load_state_dict(torch.load( self.path/'q_star_best_model.pth'))

        return train_q_star_dict
    def train_r(self, n_steps, batch_size, writer, num_workers=0,patience=5):
        dataloaders = DataLoader(self.data, batch_size=batch_size,shuffle=True, num_workers=num_workers)
        lossF = torch.nn.MSELoss()
        for epoch in tqdm(range(n_steps)):
            epoch_loss = 0
            count = 0
            mse_r_target = 0
            mse_r = 0
            epochLoss = 0
            min_loss = np.inf
            waiting = 0
            for s, a, _,_,s_next, _, _,r in dataloaders:
                count+=1
                q_next = self.q(s_next)
                n_obs = len(q_next)
                q_star_next = self.q_star.forward(s_next)
                v_next = torch.log(torch.exp(q_next).sum(1))[:,None]
                negative_log_p_star = -(q_next - v_next)[:,0][:,None]
                
                #Get adjusted q
                q = self.q(s)
                q_star = self.q_star(s)
                adj_q = (q - q[:,0][:,None])+q_star

                r_target = adj_q - self.discount*(negative_log_p_star + q_star_next)
                r_target_selected = r_target.gather(1, a.view(-1,1))
                
                r_est = self.r(s).gather(1, a.view(-1,1))
                lossR = lossF(r_est, r_target_selected)
                self.r_optimizer.zero_grad()
                lossR.backward()
                #print(lossR)
                self.r_optimizer.step()
                mse_r_target += ((r_target_selected - r)**2).mean().item()
                mse_r += ((r_est - r)**2).mean().item()
                epoch_loss +=  lossR.item()
            train_r_dict = {'Train Loss for r':epoch_loss/count,
                'MSE for r target (in practice unknonw)':mse_r_target/count,
                'MSE for r (unknown in practice)':mse_r/count
            }
            for k, value in train_r_dict.items():
                writer.add_scalar('train_r/'+k, value, epoch)
            notStop = epoch_loss < min_loss
            if notStop:
                min_loss =epoch_loss
                waiting = 0
                torch.save(self.r.state_dict(), self.path/'r_best_model.pth')
            else:
                waiting+=1
            if waiting >patience:
                break
        self.r.load_state_dict(torch.load( self.path/'r_best_model.pth'))
        return train_r_dict


    def linear_fit(self):
        dataloaders = DataLoader(self.data, batch_size=len(self.data),shuffle=True, num_workers=0)
        s = torch.FloatTensor(self.data.s)
        s_next = torch.FloatTensor(self.data.s_next)
        a = self.data.a.astype(int)
        q_next = self.q(s_next)

        q_star_next = self.q_star.forward(s_next).detach().numpy()
        v_next = torch.log(torch.exp(q_next).sum(1))[:,None]
        negative_log_p_star = -(q_next - v_next)[:,0][:,None].detach().numpy()    
        q = self.q(s).detach().numpy()
        q_star = self.q_star(s).detach().numpy()
        adj_q = (q - q[:,0][:,None])+q_star
        '''
        # Find the aggregated values for both adj_q and q_star_next
        # For adj_q
        s_agg = self.data.s_agg
        clf = NearestCentroid()
        adj_q_centroids = clf.fit(adj_q, s_agg[:,0]).centroids_
        adj_q_agg = np.concatenate(list(adj_q_centroids[label,:] for label in s_agg), 0)
        # similarly for q_star_next
        s_agg_next = self.data.s_agg_next
        clf = NearestCentroid()
        q_star_next_centroids = clf.fit(q_star_next, s_agg_next[:,0]).centroids_
        q_star_next_agg = np.concatenate(list(q_star_next_centroids[label,:] for label in s_agg_next), 0)
        r_target = (adj_q_agg - self.discount*(negative_log_p_star + q_star_next_agg))        
        '''
        r_target = (adj_q - self.discount*(negative_log_p_star + q_star_next))
        r_target_selected = np.choose(a[:,0], r_target.T)
        idx = (a!=0)[:,0]
        r_target_non_anchor = r_target_selected[idx]
        lr_model = LinearRegression()
        lr_model.fit(self.data.s_real[idx], r_target_non_anchor)
        return lr_model.coef_

    def aggregate(self, delta, aggregation_method, n_states_aggregated):
        
        assert len(self.data.s) == len(self.data.s_next), 'The number of states should be the same as number of next states.'
        s = torch.FloatTensor(np.vstack((self.data.s,self.data.s_next))) # Combine s and s_next for aggregation.       
        s_test = torch.FloatTensor(self.data.s_test)
        #n_states = len(np.unique( s,axis = 0 ))
        q_est = (self.q(s)- self.q(s)[:, 0][:,None]+ self.q_star(s)).detach().numpy()
        q_test = (self.q(s_test)- self.q(s_test)[:, 0][:,None]+ self.q_star(s_test)).detach().numpy()
        s = s.detach().numpy()
        s_test = s_test.detach().numpy()
        
        AC = AgglomerativeClustering(n_clusters=n_states_aggregated, distance_threshold = delta)
        
        if aggregation_method == 'our':
            X = q_est
            X_test = q_test
        elif aggregation_method == 'state' or aggregation_method == 'no_aggregate':
            X = s
            X_test = s_test

        else:
            raise NotImplementedError
        AC.fit(X)
        labels = AC.labels_
        # For centroids in states for pqr
        clf = NearestCentroid()
        centroids = clf.fit(s, labels).centroids_
        pred_centroids = np.concatenate(list(centroids[label,:][None,:] for label in labels), 0)
        # For labels and centroids for test
        kn = KNeighborsClassifier(n_neighbors = 100)
        kn.fit(X, AC.labels_)
        label_test = kn.predict(X_test)
        pred_centroids_test = np.concatenate(list(centroids[label,:][None,:] for label in label_test), 0)
        
        file_name = str(delta)+aggregation_method+'aggregated_s.npy'
        np.save(self.path / file_name, labels[0:int(len(labels)/2)])
        file_name = str(delta)+aggregation_method+'aggregated_s_next.npy'
        np.save(self.path / file_name, labels[int(len(labels)/2):])
        file_name = str(delta)+aggregation_method+'aggregated_s_test.npy'
        np.save(self.path / file_name, label_test)

        file_name = str(delta)+aggregation_method+'aggregated_s_value.npy'
        np.save(self.path / file_name, pred_centroids[0:int(len(labels)/2),:])
        file_name = str(delta)+aggregation_method+'aggregated_s_next_value.npy'
        np.save(self.path / file_name, pred_centroids[int(len(labels)/2):,:])
        file_name = str(delta)+aggregation_method+'aggregated_s_test_value.npy'
        np.save(self.path / file_name, pred_centroids_test)

        print(f'After aggregration, there are {len(set(labels))} states.')
        return len(set(labels))



    def get_test_likelihood(self):
        s_test = self.data.s_test
        a_test = np.load(self.log.dir/'data_generation'/'a_test.npy')

        q_test = self.q(torch.FloatTensor(s_test)).detach().numpy()
        return self.get_likelihood(q_test, a_test)

    def get_likelihood(self, q, a):

        likelihood = np.zeros(len(q))

        for i in range(len(likelihood)):
            likelihood[i] = q[i,a[i,0]] - np.log(np.exp(q[i,:]).sum())
        return likelihood.mean()


    def aggregate_all(self, delta_list, aggregation_method):
        for i in tqdm(range(len(delta_list))):
            self.aggregate(delta_list[i],aggregation_method)
    
