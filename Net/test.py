import numpy as np
import os as os
from Server import Server
from client import Client
import data_generator
import torch
import torch.autograd.variable as v

# global parameter
K = 5
# prepare model



def main():
    model_path = "model/"
    model_pool = os.listdir(model_path)
    client_num = len(model_pool)
    clients = []
    for i in range(client_num):
        c = Client(i)
        clients.append(c)

    count = 0
    for model in model_pool:
        clients[count].load_model(model_path + model)
        count += 1
    
    test(clients=clients)


def test(clients):
    # generate data
    number = 1000 * len(clients)
    upper = 120
    lower = -100
    type = 1
    datas = data_generator.generate_data(number, upper, lower, type)
    score = [0] * len(clients)
    count = 0
    for c in clients:
        correct = 0
        for data in datas:
            test_data = v(torch.Tensor(data[0:2]),dtype=torch.float32)
            label = data[2]
            pre = c.test(test_data)
            # print(pre)
            result = 0
            if pre[0] >= pre[1]:
                result = 1
            else:
                result = -1
            # print(label)
            if result == label:
                correct += 1 
            score[count] = correct/len(datas)
        count += 1

    print(score)

main()