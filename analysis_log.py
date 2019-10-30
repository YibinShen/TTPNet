#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:30:55 2018

@author: shenyibin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'logs/'
train = ['TTPNet_256.log', 'TTPNet2_256.log', 'TTPNet3_256.log',
         'TTPNet4_256.log', 'TTPNet5_256.log', 'TTPNet_Rubust_256.log',
         'TTPNet6_256.log', 'TTPNet8_256.log']
READ_DATA_PATH1 = DATA_PATH+train[0]
READ_DATA_PATH2 = DATA_PATH+train[7]

log1 = open(READ_DATA_PATH1, 'r').readlines()
log1 = log1[34:]
log2 = open(READ_DATA_PATH2, 'r').readlines()
log2 = log2[34:]

test_res1 = []
test_res2 = []

for i in range(50):
    for j in range(4):
        temp = log1[i*(4+2)+j+1]
        temp = float(str.split(temp)[-1])
        test_res1.append(temp)

test_res1 = np.asarray(test_res1).reshape(-1,4)
test_res1_mean = []
for i in range(len(test_res1)):
    test_res1_mean.append(test_res1[i].mean())

for i in range(50):
    for j in range(4):
        temp = log2[i*(4+2)+j+1]
        temp = float(str.split(temp)[-1])
        test_res2.append(temp)

test_res2 = np.asarray(test_res2).reshape(-1,4)
test_res2_mean = []
for i in range(len(test_res2)):
    test_res2_mean.append(test_res2[i].mean())

plt.plot(test_res1_mean[1:])
plt.plot(test_res2_mean[1:])