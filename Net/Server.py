import random as rd
import torch
from mimetypes import init
import numpy as np
import threading
from torch.autograd import Variable as v
from torch import nn
import time

class Server():
    def __init__(self) -> None:
        self.weight = []
        self.possibility = []
        self.clients_pool = []
        self.clients_num = None
        self.grad_pool = []
        self.epsilon = 1
        self.lamda = 1
        self.time_cost = 0
        

    def get_possibility(self, clients):
        if len(self.possibility) == 0:
            total = 0
            for client in clients:
                total += len(client.data_set)
            
            for client in clients:
                self.possibility.append(len(client.data_set)/total)

    
    def aggregate_weight(self, clients):
        start_time = time.time()
        with torch.no_grad:  # 避免纳入计算图以消耗内存
            if len(self.grad_pool) == self.clients_num:
                input_weight_after_aggregate = v(torch.zeros_like(clients[0].model.get_weight(0))) #初始化每一层的张量
                hidden_weight_after_aggregate = v(torch.zeros_like(clients[0].model.get_weight(1)))
                output_weight_after_aggregate = v(torch.zeros_like(clients[0].model.get_weight(2)))
                self.get_possibility(self.clients_pool) # 获得每个客户端的权重（FedAvg算法）
                i = 0
                j = 0
                for client in self.clients_pool:
                    if rd.random() < 0.3: # 模拟一些客户端无法参与聚合(FedAvg算法)
                        continue
                    input_weight_after_aggregate += torch.mul(client.model.get_weight(0), self.possibility[j]) #加权聚合
                    hidden_weight_after_aggregate += torch.mul(client.model.get_weight(1), self.possibility[j])
                    output_weight_after_aggregate += torch.mul(client.model.get_weight(2), self.possibility[j])
                    j += 1
                self.grad_pool = [] # 清空梯度池，等待下一轮重新算
                
                for client in self.clients_pool:
                    client.model.set_weight(0, input_weight_after_aggregate) # 设置权值
                    client.model.set_weight(1, hidden_weight_after_aggregate)
                    client.model.set_weight(2, output_weight_after_aggregate)
            end_time = time.time()
            self.time_cost += end_time-start_time

    


        