#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:21:53 2018

@author: shenyibin
"""

import os
import math
import datetime
import numpy as np
import pandas as pd

from preprocess import min_lng,max_lng,min_lat,max_lat
from preprocess import generate_date,geo_distance,generate_grid

#import sktensor
#import random
#
#import ncp
#import nonnegfac.nmf as nmf

import matplotlib.pyplot as plt

def generate_tensor_byday(date):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample2/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_Tensor/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt'):
            taxi = pd.read_csv(filename, header=0)
            
            tensor_res_columns = ['grid_id', 'time_id', 'driver_id', 'trajectory_id', 'speed', 'grid_speed']
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', 'w')
            f.write(','.join(tensor_res_columns)+'\n')
            f.close()
            basetime = int(datetime.datetime.timestamp(datetime.datetime(2013,10,int(date[-2:]))))
            
            trajectory_id1 = taxi['trajectory_id'].tolist()
            taxi['trajectory_id1'] = [trajectory_id1[0]] + trajectory_id1[:-1]
            trajectory_change = np.where(taxi['trajectory_id']!=taxi['trajectory_id1'])
            trajectory_change = [0]+ list(trajectory_change[0]) + [len(taxi)]
            
            for i in range(len(trajectory_change)-1):
                taxi_temp = taxi[trajectory_change[i]:trajectory_change[i+1]].copy()
                taxi_temp['dtimestamp'] = [0.0]+list(np.diff(taxi_temp['time_gap']))
                taxi_temp['ddis'] = [0.0]+list(np.diff(taxi_temp['dis_gap']))
                taxi_temp['grid_speed1'] = (taxi_temp['ddis']/taxi_temp['dtimestamp']).fillna(0)
                taxi_temp['grid_speed2'] = list(taxi_temp['grid_speed1'].drop(taxi_temp.index[0]))+[0.0]
                taxi_temp['grid_speed'] = (taxi_temp['grid_speed1']+taxi_temp['grid_speed2'])/2
                taxi_temp['grid_speed'].values[0] = taxi_temp['grid_speed'].values[0]*2
                taxi_temp['grid_speed'].values[-1] = taxi_temp['grid_speed'].values[-1]*2
                
                taxi_temp['time_id'] = (taxi_temp['timestamp']-basetime) // (15*60)
                tensor_res = taxi_temp[tensor_res_columns]
                
                tensor_res.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')


def generate_tensor():

    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_tensor_byday, date_pool)
#    for date in date_pool:
#        generate_tensor_byday(date)


def generate_tensor_history_byday(date):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Tensor/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_TensorHistory/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt'):
            tensor_res_columns = ['grid_id', 'time_id', 'date_id', 'speed', 'grid_speed']
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', 'w')
            f.write(','.join(tensor_res_columns)+'\n')
            f.close()
            
            date_history_pool = [str(20131000 + i) for i in range(int(date[-2:])-7,int(date[-2:]))]
            for i in range(len(date_history_pool)):
                historyname = READ_DATA_PATH+'2013-10-'+date_history_pool[i][-2:]+'.txt'
                if os.path.exists(historyname):
                    taxi_history = pd.read_csv(historyname, header=0)
                    taxi_history = taxi_history.groupby(['grid_id','time_id','trajectory_id'],as_index=False).mean()
                    taxi_history = taxi_history[['grid_id','time_id','speed', 'grid_speed']].groupby(['grid_id','time_id'],as_index=False).mean()
                    taxi_history['date_id'] = i
                    taxi_history = taxi_history[tensor_res_columns]
                    taxi_history.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')
                

def generate_tensor_history():
    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(10) as p:
        p.map(generate_tensor_history_byday, date_pool)
#    for date in date_pool:
#        generate_history_byday(date)


def generate_matrix_factorization_byday(date):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_MatrixFactorization/'
    
    filename = READ_DATA_PATH+'BeijingTaxi_Tensor/'+'2013-10-'+date[-2:]+'.txt'
    filename_history = READ_DATA_PATH+'BeijingTaxi_TensorHistory/'+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    print('Process:{}'.format(filename_history))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt'):
            matrix = pd.read_csv(filename, header=0)
            matrix = matrix.groupby(['grid_id','time_id','trajectory_id'],as_index=False).mean()
            matrix_grid_average = matrix[['grid_id','time_id','speed','grid_speed']].groupby(['grid_id','time_id'],as_index=False).mean()
            matrix_grid_average.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0)
    if os.path.exists(filename_history):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_history.txt'):
            matrix_history = pd.read_csv(filename_history, header=0)
#            matrix_history = matrix_history.groupby(['grid_id','time_id','trajectory_id','date_id'],as_index=False).mean()
            matrix_history_grid_average = matrix_history[['grid_id','time_id','speed','grid_speed']].groupby(['grid_id','time_id'],as_index=False).mean()
            matrix_history_grid_average.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_history.txt', index=0)


def generate_matrix_factorization():
    date_pool = [str(20131000 + i) for i in range(1,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_matrix_factorization_byday, date_pool)


def generate_tensor_factorization_byday(date):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_TensorFactorization/'
    
    filename = READ_DATA_PATH+'BeijingTaxi_Tensor/'+'2013-10-'+date[-2:]+'.txt'
    filename_history = READ_DATA_PATH+'BeijingTaxi_TensorHistory/'+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    print('Process:{}'.format(filename_history))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt'):
            tensor = pd.read_csv(filename, header=0)
            tensor = tensor.groupby(['grid_id','time_id','trajectory_id'],as_index=False).mean()
            tensor_grid_average = tensor[['grid_id','time_id','driver_id','speed']].groupby(['grid_id','time_id','driver_id'],as_index=False).mean()
            tensor_grid_average.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0)
    if os.path.exists(filename_history):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_history.txt'):
            tensor_history = pd.read_csv(filename_history, header=0)
            tensor_history = tensor_history.groupby(['grid_id','time_id','trajectory_id','date_id'],as_index=False).mean()
            tensor_history_grid_average = tensor_history[['grid_id','time_id','driver_id','speed']].groupby(['grid_id','time_id','driver_id'],as_index=False).mean()
            tensor_history_grid_average.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_history.txt', index=0)


def generate_tensor_factorization():
    date_pool = [str(20131000 + i) for i in range(7,32)]
    from multiprocessing import Pool

    with Pool(10) as p:
        p.map(generate_tensor_factorization_byday, date_pool)


def non_negative_matrix_completion_byday(date, r=2):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_MatrixFactorization/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_MatrixCompletion/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    filename_histroy = READ_DATA_PATH+'2013-10-'+date[-2:]+'_history.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt'):
            tensor = pd.read_csv(filename, header=0)
            tensor_history = pd.read_csv(filename_histroy, header=0)
            
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', 'w')
            tensor_res_columns = ['grid_id', 'time_id', 'speed']
            f.write(','.join(tensor_res_columns)+'\n')
            f.close()
            
            for i in range(24):
                tensor_temp = tensor[(tensor['time_id']//4)==i].copy()
                tensor_history_temp = tensor_history[(tensor_history['time_id']//4)==i].copy()
                
                if (len(tensor_temp)>0) & (len(tensor_history_temp)>0):
                    tensor_temp['tensor_time_id'] = tensor_temp['time_id'] - i*4
                    tensor_history_temp['tensor_time_id'] = tensor_history_temp['time_id'] - i*4
                    tensor_temp['tensor_date_id'] = 2
                    tensor_history_temp['tensor_date_id'] = 0
                    tensor_total_temp = pd.concat([tensor_temp, tensor_history_temp])
                    # 构建张量时需要注意下标
                    tensor_total_temp['tensor_grid_id'] = tensor_total_temp['grid_id']
                    
                    tensor_triples = tensor_total_temp[['tensor_grid_id', 'tensor_time_id', 'tensor_date_id']].values
                    tensor_values = tensor_total_temp['speed'].values
            #                tensor_values[tensor_values<1] = 0
                    X_ks = sktensor.sptensor(tuple(tensor_triples.T), tensor_values, shape=(256*256,4,3))
                    X = X_ks.toarray()
                    X2 = X_ks.toarray()
                    
                    # 生成三明治
                    X[:,:,1] = X[:,:,2]
                    for grid in range(256*256):
                        for time in range(4):
                            if X[grid,time,0] != 0 and X[grid,time,2] == 0:
                                X[grid,time,1] = X[grid,time,0]
                    X_ks_res = sktensor.dtensor(X)
                    
                    X_res = np.zeros((256*256,4,3))
                    
                    # 3 times
                    for j in range(3):
                        X_approx_ks = ncp.nonnegative_tensor_factorization(X_ks_res, r, method='anls_bpp', max_iter=50)
                        X_approx = X_approx_ks.totensor()
                        X_res[X_res<X_approx] = X_approx[X_res<X_approx]
    #                X_res = X_res/3
                    
                    X_res[X_res < 1] = 0
                    tensor_res_list = []
                    
                    for grid in range(256*256):
                        for time in range(4):
                            if X[grid,time,0]!=0 and X[grid,time,2]==0 and X_res[grid,time,1]>0:
                                tensor_res_list.append([grid,time,X_res[grid,time,1]])
                    
                    tensor_res = pd.DataFrame(tensor_res_list)
                    tensor_res.columns = tensor_res_columns
                    tensor_res['time_id'] = tensor_res['time_id'] + i*4
                    tensor_res = tensor_res[tensor_res_columns]
                    tensor_res.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')
                    
                    print('date:',date,'   i:',i)

    #                # non-negative-matrix factorization
    #                matrix_temp = tensor[(tensor['time_id']//4)==i].copy()
    #                matrix_history_temp = tensor_history[(tensor_history['time_id']//4)==i].copy()
    #                matrix_temp['matrix_time_id'] = matrix_temp['time_id'] - (i-1)*4
    #                matrix_history_temp['matrix_time_id'] = matrix_history_temp['time_id'] - i*4
    #                matrix_total_temp = pd.concat([matrix_temp, matrix_history_temp])
    #                # 构建张量时需要注意下标
    #                matrix_total_temp['matrix_grid_id'] = matrix_total_temp['grid_id'] - 1
    #                
    #                matrix_triples = matrix_total_temp[['matrix_grid_id', 'matrix_time_id']].values
    #                matrix_values = matrix_total_temp['speed'].values*1000
    #                
    #                X = np.zeros((256*256+2, 8))
    #                for i in range(len(matrix_triples)):
    #                    X[matrix_triples[i][0], matrix_triples[i][1]] = matrix_values[i]
    #                
    #                W, H, info = nmf.NMF().run(X, r)
    #                X_approx = W.dot(H.T)
    #                min_X = min(X[X!=0])
    #                X_approx[X_approx < min_X] = 0
    #                    
    #                X_err = np.linalg.norm(X-X_approx) / np.linalg.norm(X)
    #                print("Error:", X_err)


def non_negative_matrix_completion():
    date_pool = [str(20131000 + i) for i in range(2,7)]
    from multiprocessing import Pool
#
    with Pool(5) as p:
        p.map(non_negative_matrix_completion_byday, date_pool)
#    generate_non_negative_matrix_completion_byday('20131015')


def test_matrix_completion(date, i):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_MatrixFactorization/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    filename_histroy = READ_DATA_PATH+'2013-10-'+date[-2:]+'_history.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        tensor = pd.read_csv(filename, header=0)
        tensor_history = pd.read_csv(filename_histroy, header=0)
                
        # non-negative-tensor completion
        tensor_temp = tensor[(tensor['time_id']//4)==i].copy()
        tensor_history_temp = tensor_history[(tensor_history['time_id']//4)==i].copy()
        
        if (len(tensor_temp)>0) & (len(tensor_history_temp)>0):
            tensor_temp['tensor_time_id'] = tensor_temp['time_id'] - i*4
            tensor_history_temp['tensor_time_id'] = tensor_history_temp['time_id'] - i*4
            tensor_temp['tensor_date_id'] = 2
            tensor_history_temp['tensor_date_id'] = 0
            tensor_total_temp = pd.concat([tensor_temp, tensor_history_temp])
            # 构建张量时需要注意下标
            tensor_total_temp['tensor_grid_id'] = tensor_total_temp['grid_id']
            
            tensor_triples = tensor_total_temp[['tensor_grid_id', 'tensor_time_id', 'tensor_date_id']].values
            tensor_values = tensor_total_temp['speed'].values
    #                tensor_values[tensor_values<1] = 0
            X_ks = sktensor.sptensor(tuple(tensor_triples.T), tensor_values, shape=(256*256,4,3))
            X = X_ks.toarray()
            X2 = X_ks.toarray()
            
            grid_id_list = []
            time_id_list = []
            speed_list = []
            
            # 随机把一部分值变为0
            for grid in range(256*256):
                for time in range(4):
                    if X2[grid,time,0] !=0 and X2[grid,time,2] !=0:
                        if random.randint(0,9) == 0:
                            grid_id_list.append(grid)
                            time_id_list.append(time)
                            speed_list.append(X[grid,time,2])
                            X[grid,time,2] = 0
            
            # 生成三明治
            X[:,:,1] = X[:,:,2]
            for grid in range(256*256):
                for time in range(4):
                    if X[grid,time,0] != 0 and X[grid,time,2] == 0:
                        X[grid,time,1] = X[grid,time,0]
            X_ks_res = sktensor.dtensor(X)
            
            elapsed_time_list = []
            accuary_list_MAE = []
            accuary_history_list_MAE = []
            accuary_list_RMSE = []
            accuary_history_list_RMSE = []
    
            for r in range(1,8):
                X_res = np.zeros((256*256,4,3))
                speed_approx_list = []
                speed_history_list = []
    
                starttime = datetime.datetime.now()
                # 3 times
                for j in range(3):
                    X_approx_ks = ncp.nonnegative_tensor_factorization(X_ks_res, r, method='anls_bpp', max_iter=50)
                    X_approx = X_approx_ks.totensor()
                    X_res[X_res<X_approx] = X_approx[X_res<X_approx]
#                X_res = X_res/3
                endtime = datetime.datetime.now()
                elapsed_time_list.append((endtime-starttime).seconds)
    
                X_res[X_res < 1] = 0
            
                for j in range(len(speed_list)):
                    speed_approx_list.append(X_res[grid_id_list[j],time_id_list[j],1])
                
                for j in range(len(speed_list)):
                    speed_history_list.append(X2[grid_id_list[j],time_id_list[j],0])
                
                accuary = np.abs(np.array(speed_approx_list)-np.array(speed_list))
                accuary_history = np.abs(np.array(speed_history_list)-np.array(speed_list))
                
                accuary_list_MAE.append(np.mean(accuary))
                accuary_list_RMSE.append(np.sqrt(np.mean(accuary**2)))
                accuary_history_list_MAE.append(np.mean(accuary_history))
                accuary_history_list_RMSE.append(np.sqrt(np.mean(accuary_history**2)))
                
            plt.plot(accuary_list_MAE)
            plt.plot(accuary_history_list_MAE)
            plt.show()
        return elapsed_time_list,accuary_list_MAE,accuary_list_RMSE,accuary_history_list_MAE,accuary_history_list_RMSE


def test2_matrix_completion(date, r=2):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_MatrixFactorization/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    filename_histroy = READ_DATA_PATH+'2013-10-'+date[-2:]+'_history.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        tensor = pd.read_csv(filename, header=0)
        tensor_history = pd.read_csv(filename_histroy, header=0)
        
        accuary_list_MAE = []
        accuary_history_list_MAE = []
        accuary_list_RMSE = []
        accuary_history_list_RMSE = []
                
        # non-negative-tensor completion
        for i in range(24):
            tensor_temp = tensor[(tensor['time_id']//4)==i].copy()
            tensor_history_temp = tensor_history[(tensor_history['time_id']//4)==i].copy()

            if (len(tensor_temp)>0) & (len(tensor_history_temp)>0):
                tensor_temp['tensor_time_id'] = tensor_temp['time_id'] - i*4
                tensor_history_temp['tensor_time_id'] = tensor_history_temp['time_id'] - i*4
                tensor_temp['tensor_date_id'] = 2
                tensor_history_temp['tensor_date_id'] = 0
                tensor_total_temp = pd.concat([tensor_temp, tensor_history_temp])
                # 构建张量时需要注意下标
                tensor_total_temp['tensor_grid_id'] = tensor_total_temp['grid_id']
                
                tensor_triples = tensor_total_temp[['tensor_grid_id', 'tensor_time_id', 'tensor_date_id']].values
                tensor_values = tensor_total_temp['speed'].values
        #                tensor_values[tensor_values<1] = 0
                X_ks = sktensor.sptensor(tuple(tensor_triples.T), tensor_values, shape=(256*256,4,3))
                X = X_ks.toarray()
                X2 = X_ks.toarray()
                
                grid_id_list = []
                time_id_list = []
                speed_list = []
                
                # 随机把一部分值变为0
                for grid in range(256*256):
                    for time in range(4):
                        if X2[grid,time,0] !=0 and X2[grid,time,2] !=0:
                            if random.randint(0,9) == 0:
                                grid_id_list.append(grid)
                                time_id_list.append(time)
                                speed_list.append(X[grid,time,2])
                                X[grid,time,2] = 0
                
                # 生成三明治
                X[:,:,1] = X[:,:,2]
                for grid in range(256*256):
                    for time in range(4):
                        if X[grid,time,0] != 0 and X[grid,time,2] == 0:
                            X[grid,time,1] = X[grid,time,0]
                X_ks_res = sktensor.dtensor(X)
                
                X_res = np.zeros((256*256,4,3))
                speed_approx_list = []
                speed_history_list = []
                
                # 3 times
                for j in range(3):
                    X_approx_ks = ncp.nonnegative_tensor_factorization(X_ks_res, r, method='anls_bpp', max_iter=50)
                    X_approx = X_approx_ks.totensor()
                    X_res[X_res<X_approx] = X_approx[X_res<X_approx]
#                X_res = X_res/3
    
                X_res[X_res < 1] = 0
            
                for j in range(len(speed_list)):
                    speed_approx_list.append(X_res[grid_id_list[j],time_id_list[j],1])
                
                for j in range(len(speed_list)):
                    speed_history_list.append(X2[grid_id_list[j],time_id_list[j],0])
                
                accuary = np.abs(np.array(speed_approx_list)-np.array(speed_list))
                accuary_history = np.abs(np.array(speed_history_list)-np.array(speed_list))
                
                accuary_list_MAE.append(np.mean(accuary))
                accuary_list_RMSE.append(np.sqrt(np.mean(accuary**2)))
                accuary_history_list_MAE.append(np.mean(accuary_history))
                accuary_history_list_RMSE.append(np.sqrt(np.mean(accuary_history**2)))
                
            print('date:',date,'   i:',i)
        plt.plot(accuary_list_MAE)
        plt.plot(accuary_history_list_MAE)
        plt.show()
        return accuary_list_MAE,accuary_list_RMSE,accuary_history_list_MAE,accuary_history_list_RMSE


def lfm(a,k):
    '''
    来源: https://www.cnblogs.com/tbiiann/p/6535189.html
    其在计算梯度阶段只计算了矩阵中单个点的梯度，因此其代码实现中单次迭代也是
    采用三重循环来实现。算上迭代循环，整个函数中出现了4重循环，显得效率不高
    参数a：表示需要分解的评价矩阵
    参数k：分解的属性（隐变量）个数
    '''
    assert type(a) == np.ndarray
    m, n = a.shape
    alpha = 0.01
    lambda_ = 0.01
    u = np.random.rand(m,k)
    v = np.random.randn(k,n)
    for t in range(1000):
        for i in range(m):
            for j in range(n):
                if math.fabs(a[i][j]) > 1e-4:
                    err = a[i][j] - np.dot(u[i],v[:,j])
                    for r in range(k):
                        gu = err * v[r][j] - lambda_ * u[i][r]
                        gv = err * u[i][r] - lambda_ * v[r][j]
                        u[i][r] += alpha * gu
                        v[r][j] += alpha * gv
                        print(err)
    return u,v


def LFM_ed2(D, M, k, iter_times=1000, alpha=0.01, learn_rate=0.01):
    '''
    此函数实现的是最简单的 LFM 功能
    :param D: 表示需要分解的评价矩阵, type = np.ndarray
    :param k: 分解的隐变量个数
    :param iter_times: 迭代次数
    :param alpha: 正则系数
    :param learn_rate: 学习速率
    :return:  分解完毕的矩阵 U, V, 以及误差列表err_list
    '''
#    assert type(D) == np.ndarray
    m, n = D.shape  # D size = m * n
    U = np.random.randint(0, 120, (m, k))    # 为何要一个均匀分布一个正态分布？
    V = np.random.randint(0, 120, (k, n))
    err_list = []
    for t in range(iter_times):
        # 这里，对原文中公式推导我认为是推导正确的，但是循环效率太低了，可以以矩阵形式计算
        D_est = np.matmul(U, V)
        ERR = M * (D - D_est)
        U_grad = -2 * np.matmul(ERR, V.transpose()) + 2 * alpha * U
        V_grad = -2 * np.matmul(U.transpose(), ERR) + 2 * alpha * V
        U = U - learn_rate * U_grad
        V = V - learn_rate * V_grad

        ERR2 = np.multiply(ERR, ERR)
        ERR2_sum = np.sum(np.sum(ERR2))
        err_list.append(ERR2_sum)
        
        print(err_list)
    return U, V, err_list


def lfm_test(date, r=2):
    DATA_PATH = '../../../dataset/Beijing/BeijingTaxi_DeepTTE/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_MatrixFactorization/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        tensor = pd.read_csv(filename, header=0)
        for i in range(24):
            tensor_temp = tensor[(tensor['time_id']//4)==i].copy()
            D = np.zeros((256*256, 4))
            M = np.zeros((256*256, 4))
            for j in range(len(tensor_temp)):
                D[tensor_temp['grid_id'].values[j], tensor_temp['time_id'].values[j]] = tensor_temp['speed'].values[j]
                M[tensor_temp['grid_id'].values[j], tensor_temp['time_id'].values[j]] = 1.0
            
            U, V = lfm(D, 5)
#            U, V, err_list = LFM_ed2(D, M, 5)

if __name__ == '__main__':
#    generate_tensor()
#    generate_tensor_history()
#    generate_matrix_factorization()
#    generate_tensor_factorization()
#    non_negative_matrix_completion()
#    time,MAE,RMSE,MAE_history,RMSE_history = test_matrix_completion('20131008', 7)
#    rank_error = pd.DataFrame({'time':time,'MAE':MAE,'RMSE':RMSE,'MAE_history':MAE_history,'RMSE_histroy':RMSE_history})
#    rank_error.to_csv('rank_error2.csv', index=0)
    
#    MAE,RMSE,MAE_history,RMSE_history = test2_matrix_completion('20131008', 2)
#    hour_error = pd.DataFrame({'MAE':MAE,'RMSE':RMSE,'MAE_history':MAE_history,'RMSE_histroy':RMSE_history})
#    hour_error.to_csv('hour_error.csv', index=0)
    print('test')