from random import randint
from re import I
import numpy as np
import client as client
from math import sqrt
from math import sin

# 为了能够手动计算体验，只生成二维数据
# 生成数据方便分析
def generate_decision_plane(x, y, type):
    # 1: lineaer_func
    # 2: square_func
    # 3: triangle_func
    if type == 1:
        label = linear_func(x, y)
    if type == 2:
        lable = square_func(x, y)
    if type == 3:
        lable = triangle_func(x, y)
    return label

def linear_func(x, y):
    a = 3.45
    b = -4.3
    c = a * x + b * y
    if c >= 0:
        return 1
    else:
        return -1

def square_func(x, y):
    a = 1.37
    b = -2.6
    c = 0.127
    d = a * x * x + b * x + c - y
    if d >= 0:
        return 1
    return -1

def triangle_func(x, y):
    a = 0.786
    b = 1.235
    c = -3.2
    d = a * sin(x) + b*sin(y+c)
    if d >= 0:
        return 1
    return -1

def generate_data(number, upper, lower, type):
    data = []
    for i in range(number):
        x = randint(lower, upper)
        y = randint(lower, upper)
        data.append([x, y, generate_decision_plane(x, y, type)])
    return data

def allocate_data(clients):  # 给每个客户端分配数据
    number = 1000 * len(clients)
    upper = 120
    lower = -100
    type = 1
    data = generate_data(number, upper, lower, type)
    round = 0
    p = []
    p.append(0)
    for i in range(len(clients)):
        p.append(randint(0, number))
    p.sort(reverse=False)
    p.append(number)
    for client in clients:
        for i in range(p[round], p[round + 1]):
            client.database.append(data[i])
        round += 1