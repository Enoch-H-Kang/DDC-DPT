# Solve dEnv


"""
Main playground module for RL
"""
import numpy as np
import os
import datetime
import sys
import pickle
import mdptoolbox.example
from random import choices

from SAmQ.env.dEnv import dEnv

class rlRunner():
    def __init__(self, tMatrix, x0, b, c, discount):
        self.tMatrix = tMatrix
        self.x0 = x0
        self.b = b
        self.discount = discount
        self.c = c
        self.env = dEnv(tMatrix, x0, b, c)
        self.nA, _, self.nS = tMatrix.shape
        self.data = None
        self.Q = np.ones((self.nS, self.nA))
        self.r = None
        self.V = None
    def getR(self):
        self.r = np.array([[self.env.rF(i,j)  for j in range(self.nA)] for i in range(self.nS)]  )
        print(self.r)
        return 

    def getQ(self, nI):
        Q = self.Q
        for _ in range(nI):
            qSoftMax =np.log(np.exp(Q).sum(1))   # We have epsilon here!!!
            nLarge = np.zeros((self.nS, self.nA))
            for a in range(self.nA): 
                nLarge[:,a] = np.dot(np.array(self.tMatrix[a,:,:]), qSoftMax.T)
            nQ = self.r + self.discount*nLarge
            print(((nQ - Q)**2).sum())
            Q = nQ
        self.Q = Q
        print(self.tMatrix[1,:,:])
        return 
    def getData(self,nStep):
        print(self.Q)
        s = [self.env.x]    # s is longer. 
        a = []
        r = []
        for _ in range(nStep):
            # Get a
            pA = (np.exp(self.Q)/np.exp(self.Q).sum(1)[:,None])[self.env.x]
            tempA = choices(range(self.nA), pA)[0]
            
            a.append(tempA)
            r.append(self.env.step(tempA))
            
            s.append(self.env.x)
        with open("/home/ubuntu/github/cS/data/s.pkl", 'wb') as file:
            pickle.dump(s,file)
        with open("/home/ubuntu/github/cS/data/a.pkl", 'wb') as file:
            pickle.dump(a,file)
        with open("/home/ubuntu/github/cS/data/r.pkl", 'wb') as file:
            pickle.dump(r,file)
        print(np.unique(np.array(s), return_counts=True))
        print(np.unique(np.array(a), return_counts=True))
        #print(r)


    def getV(self):
        self.V = np.log(np.exp(self.Q).sum(1))
        with open("/home/ubuntu/github/cS/data/V.pkl", 'wb') as file:
            pickle.dump(self.V,file)
        print(self.V)
        return self.V


if __name__ == "__main__":
    nS = 4
    tMatrix, _ = mdptoolbox.example.rand(nS,2)
    tMatrix[:,2:,:] = 1/nS
    #tMatrix[:,2:,-1] = 1
    testEnv = rlRunner(tMatrix, 0, 0.5, -0.5, 0.8)
    testEnv.getR()
    testEnv.getQ(100)
    testEnv.getData(10000)
    testEnv.getV()