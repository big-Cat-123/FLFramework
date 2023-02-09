
from numpy import float32
import torch.nn as nn
import torch
from torch.autograd import Variable as v
from torch.nn import functional as F




class NeuralNetWork(nn.Module):
    def __init__(self, input_dimension, output_dimension) -> None:
        super(NeuralNetWork,self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(input_dimension, 4*input_dimension),
            nn.Sigmoid(),
            nn.Linear(4*input_dimension, 16),
            nn.Sigmoid(),
            nn.Linear(16, output_dimension),
            nn.Sigmoid()
        )
        
        self.model_list = ['0', '2', '4']  
        # 因为torch中把激活函数和batchNormal也算层，所以我为了描述方便我把module_list的索引提出来了
        # 需要你根据自己的模型来更改

    def set_weight(self, level, weight_outspace):  # 设置权值
        with torch.no_grad:
            self.classification._modules[self.model_list[level]].weight.data.copy_(weight_outspace)

    def get_weight(self, level):  # 返回权值，其中level是层数，输入层是第一层，输出层是最后一层，根据module_list里面的设置来取
        with torch.no_grad:  # 避免纳入计算图消耗内存
            temp = torch.zeros_like(self.classification._modules[self.model_list[level]].weight)
            temp.copy_(self.classification._modules[self.model_list[level]].weight)
            return temp

    def forward(self, x):
        logits = self.classification(x)

        return logits
    
    def save_model(self, path):
        torch.save(self.classification, "model\\" + path + ".model")


def test():
    
    model = NeuralNetWork().to(device='cpu')
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for echo in range(100):
        shape = torch.tensor([[1,2]])
        x = v(shape.to(torch.float32), requires_grad = True)
        shape = torch.tensor([[0,0]])
        y = v(torch.zeros_like(shape, dtype=torch.float32).softmax(dim=1))
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()

# test()