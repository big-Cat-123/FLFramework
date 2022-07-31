from importlib.metadata import requires
import torch
from mimetypes import init
import numpy as np
from client import Client
import noise_generator
import threading
from torch.autograd import Variable as v
from torch import nn
import noise_generator as noi

class Server():
    def __init__(self) -> None:
        self.weight = []
        self.possibility = []
        self.clients_num = 5
        self.aggregated_client = 0
        self.clients_pool = []
        self.grad_pool = []
        self.preparing = 0
        self.done = 0
        self.lock = threading.Lock()
        

    def get_possibility(self, clients):
        if len(self.possibility) == 0:
            total = 0
            for client in clients:
                total += len(client.database)
            
            for client in clients:
                self.possibility.append(len(client.database)/total)

    def aggregate_weight(self, clients):
        if len(self.grad_pool) == self.clients_num:
            input_weight_after_aggregate = v(torch.zeros_like(clients[0].get_weight(0)))
            hidden_weight_after_aggregate = v(torch.zeros_like(clients[0].get_weight(1)))
            output_weight_after_aggregate = v(torch.zeros_like(clients[0].get_weight(2)))
            self.get_possibility(self.clients_pool)
            i = 0
            j = 0
            for client in self.clients_pool:
                input_weight_after_aggregate += torch.mul(client.get_weight(0), self.possibility[j])
                hidden_weight_after_aggregate += torch.mul(client.get_weight(1), self.possibility[j])
                output_weight_after_aggregate += torch.mul(client.get_weight(2), self.possibility[j])
                j += 1
            self.grad_pool = []
            # add noise
            input_weight_after_aggregate += noi.add_noise()
            hidden_weight_after_aggregate += noi.add_noise()
            output_weight_after_aggregate += noi.add_noise()
            
            for client in self.clients_pool:
                client.set_weight(0, input_weight_after_aggregate)
                client.set_weight(1, hidden_weight_after_aggregate)
                client.set_weight(2, output_weight_after_aggregate)


    def aggregate_weight2(self, clients):
        if len(self.grad_pool) == self.clients_num:
            self.get_possibility(self.clients_pool)
            paramList = []
            for client in clients:
                param_c = []
                for parameter in client.network.parameters():
                    param_c.append(parameter)
                paramList.append(param_c)
            param_after_aggregation = []
            for j in range(len(paramList[0])):
                param_temp = 0
                for i in range(len(paramList)):
                    param_temp += torch.mul(paramList[i][j], self.possibility[i])
                param_after_aggregation.append(param_temp)

                

        '''
        weight_after_aggregate = [0] * len(clients[0].weight)
        i = 0
        j = 0
        for client in clients:
            for w in client.weight:
                weight_after_aggregate[i] = self.possibility[j] * w
                i += 1
            j += 1
        self.weight = weight_after_aggregate
        '''
    
    def aggregate_grad(self):
        '''
        while(self.signal > 0):
            print("aggregating")
            flag = True
            for client in self.clients_pool:
                if new_client.clientID == client.clientID:
                    flag = False
            if flag:
                self.clients_pool.append(new_client)
                self.grad_pool.append(grad)
            
            while(len(self.clients_pool)!=self.clients_num):
                print(len(self.clients_pool))
                self.aggregating = False
        '''
        if len(self.grad_pool) == self.clients_num:
            grad_after_aggregate = torch.tensor(np.array([[0, 0]] * len(self.grad_pool[0])), dtype=torch.float32)
            self.get_possibility(self.clients_pool)
            i = 0
            j = 0
            for client in self.clients_pool:
                for w in self.grad_pool:
                    # temp = [self.possibility[j]] * 2
                    grad_after_aggregate += torch.mul(w, self.possibility[j])
                    i += 1
                j += 1
            self.grad_pool = []
            return grad_after_aggregate



    
    def self_broadcast(self, clients):
        for client in clients:
            client.weight = self.weight
    
    def add_noise(self):
        self.weight = noise_generator.noise(self.weight)

        