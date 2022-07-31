
import numpy as np
from random import randint
import neural_net_work as net
from torch.autograd import Variable as v
import torch 
import threading
import torch.nn as nn


class Client():
    def __init__(self, ID) -> None:
        # self.lable = randint()%2
        self.database = []
        # self.weight = []
        self.gradient = []
        self.echo = 100
        self.clientID = ID
        self.network = net.NeuralNetWork().to(device='cpu')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01)
        self.pred = 0
        self.sum_loss = 0

    def noise_generator():
        noise = 0
        return noise

    def learning_algorithm(self):
        # self.network = net.NeuralNetWork().to(device='cpu')
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01)
        # self.load_model("model/" + str(self.clientID) + ".model")
        self.sum_loss = 0
        for item in self.database:
            sample = v(torch.tensor(np.array([item[0:2]]),dtype=torch.float32), requires_grad = True)
            if item[2] == 1:
                label = v(torch.tensor(np.array([[1,0]]), dtype = torch.float32))
            else:
                label = v(torch.tensor(np.array([[0,1]]), dtype = torch.float32))
            self.pred = self.network(sample)         
            loss = self.loss_fn(self.pred, label)
            self.sum_loss += loss
        self.optimizer.zero_grad()
        self.sum_loss.backward()
        self.optimizer.step()

                
    def optimize_step(self, gradient_after_aggregation):
        self.gradient = gradient_after_aggregation
        self.pred = self.gradient
        
        self.optimizer.step()
    
    def save_model(self):
        torch.save(self.network, "model/" + str(self.clientID) + ".model")

    def test(self, test_data):
        pre = self.network(test_data)
        return pre
    
    def load_model(self, path):
        self.network = torch.load(path)
    
    
    def set_weight(self, level, weight_outspace):
        self.network.set_weight(level=level, weight_outspace=weight_outspace)


    def get_weight(self, level):
        return self.network.get_weight(level=level)

        