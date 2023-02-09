from CVSOperator import CSVOperator
import numpy as np
import random as rd
import torch
import time
import torch.nn as nn
from Net.neural_net_work import NeuralNetWork
from torch.utils.data import DataLoader

def CrossEntropy_derivative(pred, label):
    if pred == 0:
        pred = 0.00001
    pred = (1/pred)*label - (1-label)/(1-pred)
    return -pred

class Client:
    def __init__(self, data_set, ID, model_path):
        self.client_id = ID
        self.path = data_set
        self.model_path = model_path
        self.csv_operator = CSVOperator(data_set, 'r')
        self.data_set, self.test_data_set = self.load_data()
        self.model = NeuralNetWork(input_dimension=len(self.data_set[0]), output_dimension=2)


    
    def train_local_model(self):
        self.local_model.model.learning_process(self.data_set)

    def save_model(self):
        self.model.save_model(self.client_id)
    
    def training_process(self):  # 训练过程
        echo = 10
        loss_sum = 0
        loss_fn = nn.BCELoss()  
        opimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        flag = False  # 用于标记是否训练完
        for e in range(echo):
            data = DataLoader(self.data_set, batch_size=30)
            for data_batch, label_batch in data:
                sample_ = torch.tensor(data_batch, dtype=torch.float32)
                label_ = torch.tensor(label_batch, dtype=torch.float32).unsqueeze(-1)
                pred = self.model(sample_)
                loss = loss_fn(pred, label_)
                loss_sum += loss.item()
                opimizer.zero_grad()
                loss.backward()
                opimizer.step()

        flag = True
        return flag



    def test(self, e):  # 测试模型
        row = []
        title = ["+", "-", "groundtrueth"]
        time_cost_start = time.time()
        row.append(title)
        c = 0
        length = len(self.test_data_set[0])  # 我组织数据的时候标签放在第一个，可以根据自己需要调整
        for i in range(len(self.test_data_set)):
            sample_ = torch.tensor(self.test_data_set[i][1:length+1], dtype=torch.float32)
            res = self.model(sample_)
            res = res.detach().numpy()[0]
            row.append([res, 1-res, self.test_data_set[i][0]])
            c += 1
        time_cost_end = time.time()
        self.time_cost = time_cost_end-time_cost_start
        csv_writer = CSVOperator("result/"+ str(e)+self.model_path, 'w')
        csv_writer.write_row(row)
            
            

    def load_data(self):
        data_set_train = []
        data_set_test = []
        data_set_all = []
        c = 0
        for row in self.csv_operator.reader:
            if c == 0:
                c += 1
                continue
            data_set_all.append(row)
        rd.seed(self.client_id) # 保证每一次随机的数据集都是一样的，因为论文实验需要可重复性，也方便分析
        rd.shuffle(data_set_all)
        for row in data_set_all:
            if c < np.floor(len(data_set_all)*0.7):
                data_set_train.append(np.array(row))
            else:
                data_set_test.append(np.array(row))
            c += 1
        
        return data_set_train,  data_set_test


def data_format_conversion(data_row):  # array转tensor
        data_row = np.array(data_row)
        data_input = torch.tensor(data_row, dtype=torch.float32)
        return data_input