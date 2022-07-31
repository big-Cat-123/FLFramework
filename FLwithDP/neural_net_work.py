
from importlib.metadata import requires
import re
from numpy import float32
import torch.nn as nn
import torch
from torch.autograd import Variable as v
from torch.nn import functional as F
import noise_generator as noi



class NeuralNetWork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetWork,self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2)
        )
        self.model_list = ['0', '2', '4']

    
    def set_weight(self, level, weight_outspace):
        self.classification._modules[self.model_list[level]].weight.data = nn.Parameter(weight_outspace)


    def get_weight(self, level):
        return self.classification._modules[self.model_list[level]].weight + noi.add_noise()

    def forward(self, x):
        logits = self.classification(x)

        return logits


def test():
    
    model = NeuralNetWork().to(device='cpu')
    loss_fn = torch.nn.CrossEntropyLoss()
    # print(model.classification._modules['0'].weight)
    # model.classification._modules['0'].weight = nn.Parameter(torch.zeros(2, 10).to(torch.device('cpu')))
    # print(model.classification._modules['0'].weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # print(model.parameters)

    sig = nn.Sigmoid()
    for echo in range(100):
        shape = torch.tensor([[1,2]])
        x = v(shape.to(torch.float32), requires_grad = True)
        # shape = torch.tensor([1])
        shape = torch.tensor([[0,0]])
        y = v(torch.zeros_like(shape, dtype=torch.float32).softmax(dim=1))
        pred = model(x)
        # print(pred.shape)
        # print(y.shape)
        loss = loss_fn(pred, y)
        # print(pred.grad)
        optimizer.zero_grad()
        # print(model.classification._modules['0'].weight.grad)
        loss.backward()
        
        # print(loss)
        # print(model.classification._modules['0'].weight.grad)
        optimizer.step()
    # print(model.classification._modules['0'].weight)

# test()