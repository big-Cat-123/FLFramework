import numpy as np
import random as rd

def laplace_noise():
    return 0

def guassian_noise():
    return 0

def add_noise():
    noise = laplace_noise()
    return noise

def add_laplace_noise(length, miu, epsilon, lam):
    b = lam/epsilon
    laplace_noise = np.random.laplace(loc=0, scale=b, size=length)
    return laplace_noise

def add_gaussian_noise(datalist, miu, epsilon):
    gaussian_noise = []
    delta = 10e-5
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / epsilon
    for i in range(len(datalist)):
        gaussian_noise.append(rd.gauss(miu, sigma=sigma))
    return gaussian_noise