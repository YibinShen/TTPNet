#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:30:29 2019

@author: shenyibin
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from preprocess import min_lng,max_lng,min_lat,max_lat
from preprocess import generate_date,geo_distance,generate_grid


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def generate_iTCL_Train_Forward_byday(date, time_interval=4):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Train1/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Forward/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
    print('Process:{}'.format(filename))
    
    if date == '20131015':
        date_yesterday = '20131013'
    else:
        date_yesterday = str(int(date)-1)
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi = open(filename,'r').readlines()
            taxi = list(map(lambda x:json.loads(x), taxi))
            
            matrix_today = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date[-2:]+'.txt', header=0)
            matrix_yesterday = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date_yesterday[-2:]+'.txt', header=0)
            #yesterday只能使用23~24点的数据
            matrix_yesterday = matrix_yesterday[matrix_yesterday['time_id']//4==23]
            matrix_history = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date[-2:]+'_history.txt', header=0)
            
            matrix_today_completion = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixCompletion/'+'2013-10-'+date[-2:]+'.txt', header=0)
            matrix_yesterday_completion = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixCompletion/'+'2013-10-'+date_yesterday[-2:]+'.txt', header=0)
            #yesterday只能使用23~24点的数据
            matrix_yesterday_completion = matrix_yesterday_completion[matrix_yesterday_completion['time_id']//4==23]
            
            matrix_today_total = matrix_today[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
            matrix_yesterday_total = matrix_yesterday[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
            matrix_history_total = matrix_history[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
            
            matrix_today_array = np.zeros((256*256,96))-1
            matrix_yesterday_array = np.zeros((256*256,96))-1
            matrix_history_array = np.zeros((256*256,96))-1
            matrix_today_completion_array = np.zeros((256*256,96))-1
            matrix_yesterday_completion_array = np.zeros((256*256,96))-1
            
            for i in range(len(matrix_today)):
                matrix_today_array[matrix_today['grid_id'].values[i],
                                   matrix_today['time_id'].values[i]] = matrix_today['speed'].values[i]
            for i in range(len(matrix_history)):
                matrix_history_array[matrix_history['grid_id'].values[i],
                                     matrix_history['time_id'].values[i]] = matrix_history['speed'].values[i]
            if len(matrix_yesterday):
                for i in range(len(matrix_yesterday)):
                    matrix_yesterday_array[matrix_yesterday['grid_id'].values[i],
                                           matrix_yesterday['time_id'].values[i]] = matrix_yesterday['speed'].values[i]
            
            for i in range(len(matrix_today_completion)):
                matrix_today_completion_array[matrix_today_completion['grid_id'].values[i],
                                              matrix_today_completion['time_id'].values[i]] = matrix_today_completion['speed'].values[i]
            if len(matrix_yesterday_completion):
                for i in range(len(matrix_yesterday_completion)):
                    matrix_yesterday_completion_array[matrix_yesterday_completion['grid_id'].values[i],
                                                      matrix_yesterday_completion['time_id'].values[i]] = matrix_yesterday_completion['speed'].values[i]

            for i in range(len(taxi)):
                temp = taxi[i].copy()
                grid_id_list = temp['grid_id'].copy()
                speeds_forward_list = []
                # 过去的时间，这很重要
                time_id_tensor = temp['timeID']//15-time_interval
                time_id_tensor_pool = [time_id_tensor+j for j in range(time_interval)]
                
                for k in range(len(grid_id_list)):
                    for time_id in time_id_tensor_pool:
                        # 要用昨天的数据
                        if time_id <= -1:
                            time_id += 96
                            grid_speed_yesterday = matrix_yesterday_array[grid_id_list[k], time_id]
                            if grid_speed_yesterday >= 0:
                                speeds_forward_list.append(grid_speed_yesterday)
                            else:
                                grid_speed_yesterday_completion = matrix_yesterday_completion_array[grid_id_list[k], time_id]
                                if grid_speed_yesterday_completion >= 0:
                                    speeds_forward_list.append(grid_speed_yesterday_completion)
                                else:
                                    grid_speed_history = matrix_history_array[grid_id_list[k], time_id]
                                    if grid_speed_history >= 0:
                                        speeds_forward_list.append(grid_speed_history)
                                    else:
                                        grid_speed_yesterday_total = matrix_yesterday_total[matrix_yesterday_total['time_id']==time_id]['speed']
                                        grid_speed_history_total = matrix_history_total[matrix_history_total['time_id']==time_id]['speed']
                                        if len(grid_speed_yesterday_total) > 0:
                                            speeds_forward_list.append(grid_speed_yesterday_total.values[0])
                                        else:
                                            speeds_forward_list.append(grid_speed_history_total.values[0])
                        # 要用今天的数据
                        else:
                            grid_speed_today = matrix_today_array[grid_id_list[k], time_id]
                            if grid_speed_today >= 0:
                                speeds_forward_list.append(grid_speed_today)
                            else:
                                grid_speed_today_completion = matrix_today_completion_array[grid_id_list[k], time_id]
                                if grid_speed_today_completion >= 0:
                                    speeds_forward_list.append(grid_speed_today_completion)
                                else:
                                    grid_speed_history = matrix_history_array[grid_id_list[k], time_id]
                                    if grid_speed_history >= 0:
                                        speeds_forward_list.append(grid_speed_history)
                                    else:
                                        grid_speed_today_total = matrix_today_total[matrix_today_total['time_id']==time_id]['speed']
                                        grid_speed_history_total = matrix_history_total[matrix_history_total['time_id']==time_id]['speed']
                                        if len(grid_speed_today_total) > 0:
                                            speeds_forward_list.append(grid_speed_today_total.values[0])
                                        else:
                                            speeds_forward_list.append(grid_speed_history_total.values[0])

#                speeds_forward = np.asarray(speeds_forward_list).reshape(-1, time_interval)
                temp['speeds_forward'] = speeds_forward_list
                json.dump(temp, f, cls=MyEncoder)
                f.write('\n')
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi))
            f.close()


def generate_iTCL_Train_Forward():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_Forward_byday, date_pool)
    

def generate_iTCL_Train_History_byday(date, dayinterval=7):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Train1/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_History/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi = open(filename,'r').readlines()
            taxi = list(map(lambda x:json.loads(x), taxi))

            matrix_history = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date[-2:]+'_history.txt', header=0)
            matrix_history_total = matrix_history[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
            matrix_history_array = np.zeros((256*256,96))-1
            
            for i in range(len(matrix_history)):
                matrix_history_array[matrix_history['grid_id'].values[i],
                                     matrix_history['time_id'].values[i]] = matrix_history['speed'].values[i]
        
            date_history_pool = [str(20131000 + i) for i in range(int(date[-2:])-dayinterval,int(date[-2:]))]
            for i in range(len(date_history_pool)):
                if date_history_pool[i] == '20131014':
                    date_history_pool[i] = '20131013'
            matrix_array = np.zeros((256*256,96,7))-1
            # 注意这里与之前的不同
            matrix_total_array = np.zeros((96, 7))-1
            matrix_completion_array = np.zeros((256*256,96,7))-1
            
            for i in range(len(date_history_pool)):
                historyname = DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date_history_pool[i][-2:]+'.txt'
                historyname_completion = DATA_PATH+'BeijingTaxi_MatrixCompletion/'+'2013-10-'+date_history_pool[i][-2:]+'.txt'
                if os.path.exists(historyname):
                    matrix = pd.read_csv(historyname, header=0)
                    matrix_total = matrix[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
                    for j in range(len(matrix)):
                        matrix_array[matrix['grid_id'].values[j],
                                     matrix['time_id'].values[j], i] = matrix['speed'].values[j]
                    for j in range(len(matrix_total)):
                        matrix_total_array[matrix_total['time_id'].values[j], i] = matrix_total['speed'].values[j]
                
                if os.path.exists(historyname_completion):
                    matrix_completion = pd.read_csv(historyname_completion, header=0)
                    for j in range(len(matrix_completion)):
                        matrix_completion_array[matrix_completion['grid_id'].values[j],
                                                matrix_completion['time_id'].values[j], i] = matrix_completion['speed'].values[j]
            
            for i in range(len(taxi)):
                temp = taxi[i].copy()
                grid_id_list = temp['grid_id'].copy()
                speeds_history_list = [] 
                time_id_tensor = temp['timeID']//15
                
                for j in range(len(grid_id_list)):
                    for day_id_tensor in range(dayinterval):
                        grid_speed = matrix_array[grid_id_list[j], time_id_tensor, day_id_tensor]
                        if grid_speed >= 0:
                            speeds_history_list.append(grid_speed)
                        else:
                            grid_speed_completion = matrix_completion_array[grid_id_list[j], time_id_tensor, day_id_tensor]
                            if grid_speed_completion >= 0:
                                speeds_history_list.append(grid_speed_completion)
                            else:
                                grid_speed_history = matrix_history_array[grid_id_list[j], time_id_tensor]
                                if grid_speed_history >= 0:
                                    speeds_history_list.append(grid_speed_history)
                                else:
                                    # 注意这里与之前的不同
                                    grid_speed_total = matrix_total_array[time_id_tensor, day_id_tensor]
                                    if grid_speed_total >= 0:
                                        speeds_history_list.append(grid_speed_total)
                                    else:
                                        grid_speed_history_total = matrix_history_total[matrix_history_total['time_id']==time_id_tensor]['speed']
                                        speeds_history_list.append(grid_speed_history_total.values[0])
                
#                speeds_history = np.asarray(speeds_history_list).reshape(-1, day_interval)
                temp['speeds_history'] = speeds_history_list
                json.dump(temp, f, cls=MyEncoder)
                f.write('\n')
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi))
            f.close()


def generate_iTCL_Train_History():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_History_byday, date_pool)


def generate_iTCL_Train_Adjacent_byday(date, time_interval=4):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Train1/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Adjacent/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
    print('Process:{}'.format(filename))
    
    adjacent_grid1_file = open(DATA_PATH + 'adjacent_grid1.json', 'r')
    adjacent_grid1_dict = json.loads(adjacent_grid1_file.read())
    adjacent_grid1_file.close()
    adjacent_grid2_file = open(DATA_PATH + 'adjacent_grid2.json', 'r')
    adjacent_grid2_dict = json.loads(adjacent_grid2_file.read())
    adjacent_grid2_file.close()
    
    if date == '20131015':
        date_yesterday = '20131013'
    else:
        date_yesterday = str(int(date)-1)

    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi = open(filename,'r').readlines()
            taxi = list(map(lambda x:json.loads(x), taxi))
            
            matrix_today = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date[-2:]+'.txt', header=0)
            matrix_yesterday = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date_yesterday[-2:]+'.txt', header=0)
            #yesterday只能使用23~24点的数据
            matrix_yesterday = matrix_yesterday[matrix_yesterday['time_id']//4==23]
            matrix_history = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixFactorization/'+'2013-10-'+date[-2:]+'_history.txt', header=0)
            
            matrix_today_completion = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixCompletion/'+'2013-10-'+date[-2:]+'.txt', header=0)
            matrix_yesterday_completion = pd.read_csv(DATA_PATH+'BeijingTaxi_MatrixCompletion/'+'2013-10-'+date_yesterday[-2:]+'.txt', header=0)
            #yesterday只能使用23~24点的数据
            matrix_yesterday_completion = matrix_yesterday_completion[matrix_yesterday_completion['time_id']//4==23]
            
            matrix_today_total = matrix_today[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
            matrix_yesterday_total = matrix_yesterday[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
            matrix_history_total = matrix_history[['time_id','speed']].groupby(['time_id'],as_index=False).mean()
            
            matrix_today_array = np.zeros((256*256,96))-1
            matrix_yesterday_array = np.zeros((256*256,96))-1
            matrix_history_array = np.zeros((256*256,96))-1
            matrix_today_completion_array = np.zeros((256*256,96))-1
            matrix_yesterday_completion_array = np.zeros((256*256,96))-1
            
            for i in range(len(matrix_today)):
                matrix_today_array[matrix_today['grid_id'].values[i],
                                   matrix_today['time_id'].values[i]] = matrix_today['speed'].values[i]
            for i in range(len(matrix_history)):
                matrix_history_array[matrix_history['grid_id'].values[i],
                                     matrix_history['time_id'].values[i]] = matrix_history['speed'].values[i]
            if len(matrix_yesterday):
                for i in range(len(matrix_yesterday)):
                    matrix_yesterday_array[matrix_yesterday['grid_id'].values[i],
                                           matrix_yesterday['time_id'].values[i]] = matrix_yesterday['speed'].values[i]
            
            for i in range(len(matrix_today_completion)):
                matrix_today_completion_array[matrix_today_completion['grid_id'].values[i],
                                              matrix_today_completion['time_id'].values[i]] = matrix_today_completion['speed'].values[i]
            if len(matrix_yesterday_completion):
                for i in range(len(matrix_yesterday_completion)):
                    matrix_yesterday_completion_array[matrix_yesterday_completion['grid_id'].values[i],
                                                      matrix_yesterday_completion['time_id'].values[i]] = matrix_yesterday_completion['speed'].values[i]
            
            for i in range(len(taxi)):
                temp = taxi[i].copy()
                grid_id_list = temp['grid_id'].copy()
                adjacent_grid1_list = []
                adjacent_grid2_list = []
                
                for j in range(len(grid_id_list)):
                    if str(grid_id_list[j]) in adjacent_grid1_dict:
                        adjacent_grid1_list.append(adjacent_grid1_dict[str(grid_id_list[j])])
                    else:
                        adjacent_grid1_list.append(grid_id_list[j])
                for j in range(len(grid_id_list)):
                    if str(grid_id_list[j]) in adjacent_grid2_dict:
                        adjacent_grid2_list.append(adjacent_grid2_dict[str(grid_id_list[j])])
                    else:
                        adjacent_grid2_list.append(grid_id_list[j])
                
                speeds_adjacent1_list = []
                speeds_adjacent2_list = []
                
                # 过去的时间，这很重要
                time_id_tensor = temp['timeID']//15-time_interval
                time_id_tensor_pool = [time_id_tensor+j for j in range(time_interval)]
                
                for k in range(len(adjacent_grid1_list)):
                    for time_id in time_id_tensor_pool:
                        # 要用昨天的数据
                        if time_id <= -1:
                            time_id += 96
                            grid_speed1_yesterday = matrix_yesterday_array[adjacent_grid1_list[k], time_id]
                            if grid_speed1_yesterday >= 0:
                                speeds_adjacent1_list.append(grid_speed1_yesterday)
                            else:
                                grid_speed1_yesterday_completion = matrix_yesterday_completion_array[adjacent_grid1_list[k], time_id]
                                if grid_speed1_yesterday_completion >= 0:
                                    speeds_adjacent1_list.append(grid_speed1_yesterday_completion)
                                else:
                                    grid_speed1_history = matrix_history_array[adjacent_grid1_list[k], time_id]
                                    if grid_speed1_history >= 0:
                                        speeds_adjacent1_list.append(grid_speed1_history)
                                    else:
                                        grid_speed1_yesterday_total = matrix_yesterday_total[matrix_yesterday_total['time_id']==time_id]['speed']
                                        grid_speed1_history_total = matrix_history_total[matrix_history_total['time_id']==time_id]['speed']
                                        if len(grid_speed1_yesterday_total) > 0:
                                            speeds_adjacent1_list.append(grid_speed1_yesterday_total.values[0])
                                        else:
                                            speeds_adjacent1_list.append(grid_speed1_history_total.values[0])
                        # 要用今天的数据
                        else:
                            grid_speed1_today = matrix_today_array[adjacent_grid1_list[k], time_id]
                            if grid_speed1_today >= 0:
                                speeds_adjacent1_list.append(grid_speed1_today)
                            else:
                                grid_speed1_today_completion = matrix_today_completion_array[adjacent_grid1_list[k], time_id]
                                if grid_speed1_today_completion >= 0:
                                    speeds_adjacent1_list.append(grid_speed1_today_completion)
                                else:
                                    grid_speed1_history = matrix_history_array[adjacent_grid1_list[k], time_id]
                                    if grid_speed1_history >= 0:
                                        speeds_adjacent1_list.append(grid_speed1_history)
                                    else:
                                        grid_speed1_today_total = matrix_today_total[matrix_today_total['time_id']==time_id]['speed']
                                        grid_speed1_history_total = matrix_history_total[matrix_history_total['time_id']==time_id]['speed']
                                        if len(grid_speed1_today_total) > 0:
                                            speeds_adjacent1_list.append(grid_speed1_today_total.values[0])
                                        else:
                                            speeds_adjacent1_list.append(grid_speed1_history_total.values[0])
                
                
                for k in range(len(adjacent_grid2_list)):
                    for time_id in time_id_tensor_pool:
                        # 要用昨天的数据
                        if time_id <= -1:
                            time_id += 96
                            grid_speed2_yesterday = matrix_yesterday_array[adjacent_grid2_list[k], time_id]
                            if grid_speed2_yesterday >= 0:
                                speeds_adjacent2_list.append(grid_speed2_yesterday)
                            else:
                                grid_speed2_yesterday_completion = matrix_yesterday_completion_array[adjacent_grid2_list[k], time_id]
                                if grid_speed2_yesterday_completion >= 0:
                                    speeds_adjacent2_list.append(grid_speed2_yesterday_completion)
                                else:
                                    grid_speed2_history = matrix_history_array[adjacent_grid2_list[k], time_id]
                                    if grid_speed2_history >= 0:
                                        speeds_adjacent2_list.append(grid_speed2_history)
                                    else:
                                        grid_speed2_yesterday_total = matrix_yesterday_total[matrix_yesterday_total['time_id']==time_id]['speed']
                                        grid_speed2_history_total = matrix_history_total[matrix_history_total['time_id']==time_id]['speed']
                                        if len(grid_speed2_yesterday_total) > 0:
                                            speeds_adjacent2_list.append(grid_speed2_yesterday_total.values[0])
                                        else:
                                            speeds_adjacent2_list.append(grid_speed2_history_total.values[0])
                        # 要用今天的数据
                        else:
                            grid_speed2_today = matrix_today_array[adjacent_grid2_list[k], time_id]
                            if grid_speed2_today >= 0:
                                speeds_adjacent2_list.append(grid_speed2_today)
                            else:
                                grid_speed2_today_completion = matrix_today_completion_array[adjacent_grid2_list[k], time_id]
                                if grid_speed2_today_completion >= 0:
                                    speeds_adjacent2_list.append(grid_speed2_today_completion)
                                else:
                                    grid_speed2_history = matrix_history_array[adjacent_grid2_list[k], time_id]
                                    if grid_speed2_history >= 0:
                                        speeds_adjacent2_list.append(grid_speed2_history)
                                    else:
                                        grid_speed2_today_total = matrix_today_total[matrix_today_total['time_id']==time_id]['speed']
                                        grid_speed2_history_total = matrix_history_total[matrix_history_total['time_id']==time_id]['speed']
                                        if len(grid_speed2_today_total) > 0:
                                            speeds_adjacent2_list.append(grid_speed2_today_total.values[0])
                                        else:
                                            speeds_adjacent2_list.append(grid_speed2_history_total.values[0])
                
                temp['speeds_adjacent1'] = speeds_adjacent1_list
                temp['speeds_adjacent2'] = speeds_adjacent2_list
                json.dump(temp, f, cls=MyEncoder)
                f.write('\n')
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi))
            f.close()


def generate_iTCL_Train_Adjacent():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_Adjacent_byday, date_pool)


def generate_iTCL_Train_DeepWalk_byday(date, dim=16):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Train1/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_DeepWalk/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
    print('Process:{}'.format(filename))
    
    grid_id_file = open(DATA_PATH + 'grid_id_mapping.json', 'r')
    grid_id_dict = json.loads(grid_id_file.read())
    grid_id_file.close()
    
    embedding = pd.read_csv(DATA_PATH+'Beijing_deepwalk.embeddings', sep=' ', header=-1, index_col=0)
    matrix_embedding = np.zeros((256*256,dim))
    for i in range(256*256):
        if str(i) in grid_id_dict:
            matrix_embedding[i] = embedding.loc[grid_id_dict[str(i)]].values

    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi = open(filename,'r').readlines()
            taxi = list(map(lambda x:json.loads(x), taxi))
            
            for i in range(len(taxi)):
                temp = taxi[i].copy()
                grid_id_list = temp['grid_id'].copy()
                deepwalk_embedding_list = []
                
                for j in range(len(grid_id_list)):
                    deepwalk_embedding_list += matrix_embedding[grid_id_list[j]].tolist()
                
                temp['deepwalk_embedding'] = deepwalk_embedding_list
                json.dump(temp, f)
                f.write('\n')
                
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi))
            f.close()


def generate_iTCL_Train_DeepWalk():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_DeepWalk_byday, date_pool)


def generate_iTCL_Train_SDNE_byday(date, dim=16):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Train1/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_SDNE/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
    print('Process:{}'.format(filename))
    
    grid_id_file = open(DATA_PATH + 'grid_id_mapping.json', 'r')
    grid_id_dict = json.loads(grid_id_file.read())
    grid_id_file.close()
    
    embedding = sio.loadmat(DATA_PATH+'Beijing_SDNE_embeddings.mat')
    matrix_embedding = np.zeros((256*256,dim))
    for i in range(256*256):
        if str(i) in grid_id_dict:
            matrix_embedding[i] = embedding['embedding'][grid_id_dict[str(i)]]

    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi = open(filename,'r').readlines()
            taxi = list(map(lambda x:json.loads(x), taxi))
            
            for i in range(len(taxi)):
                temp = taxi[i].copy()
                grid_id_list = temp['grid_id'].copy()
                SDNE_embedding_list = []
                
                for j in range(len(grid_id_list)):
                    SDNE_embedding_list += matrix_embedding[grid_id_list[j]].tolist()
                
                temp['SDNE_embedding'] = SDNE_embedding_list
                json.dump(temp, f)
                f.write('\n')
                
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi))
            f.close()


def generate_iTCL_Train_SDNE():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_SDNE_byday, date_pool)


def generate_iTCL_Train_Whole1_byday(date):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH1 = DATA_PATH+'BeijingTaxi_iTCL_Train_Forward/'
    READ_DATA_PATH2 = DATA_PATH+'BeijingTaxi_iTCL_Train_History/'
    READ_DATA_PATH3 = DATA_PATH+'BeijingTaxi_iTCL_Train_Adjacent/'
    READ_DATA_PATH4 = DATA_PATH+'BeijingTaxi_STDR_Train_Road/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Whole1/'
    
    filename1 = READ_DATA_PATH1+'2013-10-'+date[-2:]+'.json'
    filename2 = READ_DATA_PATH2+'2013-10-'+date[-2:]+'.json'
    filename3 = READ_DATA_PATH3+'2013-10-'+date[-2:]+'.json'
    filename4 = READ_DATA_PATH4+'2013-10-'+date[-2:]+'.json'
    
    print('Process:{}'.format(filename1))
    print('Process:{}'.format(filename2))
    print('Process:{}'.format(filename3))
    print('Process:{}'.format(filename4))
    
    if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3) and os.path.exists(filename4):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi1 = open(filename1,'r').readlines()
            taxi1 = list(map(lambda x:json.loads(x), taxi1))
            taxi2 = open(filename2,'r').readlines()
            taxi2 = list(map(lambda x:json.loads(x), taxi2))
            taxi3 = open(filename3,'r').readlines()
            taxi3 = list(map(lambda x:json.loads(x), taxi3))
            taxi4 = open(filename4,'r').readlines()
            taxi4 = list(map(lambda x:json.loads(x), taxi4))
            
            for i in range(len(taxi1)):
                temp1 = taxi1[i].copy()
                temp2 = taxi2[i].copy()
                temp3 = taxi3[i].copy()
                temp4 = taxi4[i].copy()
                
                driverID = np.unique([temp1['driverID'], temp2['driverID'],
                                      temp3['driverID'], temp4['driverID']])
                
                if len(driverID) <= 1:
                    temp1['speeds_history'] = temp2['speeds_history']
                    temp1['speeds_adjacent1'] = temp3['speeds_adjacent1']
                    temp1['speeds_adjacent2'] = temp3['speeds_adjacent2']
                    temp1['road_embedding'] = temp4['road_embedding']
                else:
                    print('date:',date,'   error:',i)
                
                json.dump(temp1, f, cls=MyEncoder)
                f.write('\n')
                
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi1))
            f.close()


def generate_iTCL_Train_Whole1():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_Whole1_byday, date_pool)


def generate_iTCL_Train_Whole2_byday(date):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH1 = DATA_PATH+'BeijingTaxi_iTCL_Train_Forward/'
    READ_DATA_PATH2 = DATA_PATH+'BeijingTaxi_iTCL_Train_History/'
    READ_DATA_PATH3 = DATA_PATH+'BeijingTaxi_iTCL_Train_Adjacent/'
    READ_DATA_PATH4 = DATA_PATH+'BeijingTaxi_iTCL_Train_DeepWalk/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Whole2/'
    
    filename1 = READ_DATA_PATH1+'2013-10-'+date[-2:]+'.json'
    filename2 = READ_DATA_PATH2+'2013-10-'+date[-2:]+'.json'
    filename3 = READ_DATA_PATH3+'2013-10-'+date[-2:]+'.json'
    filename4 = READ_DATA_PATH4+'2013-10-'+date[-2:]+'.json'
    
    print('Process:{}'.format(filename1))
    print('Process:{}'.format(filename2))
    print('Process:{}'.format(filename3))
    print('Process:{}'.format(filename4))
    
    if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3) and os.path.exists(filename4):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi1 = open(filename1,'r').readlines()
            taxi1 = list(map(lambda x:json.loads(x), taxi1))
            taxi2 = open(filename2,'r').readlines()
            taxi2 = list(map(lambda x:json.loads(x), taxi2))
            taxi3 = open(filename3,'r').readlines()
            taxi3 = list(map(lambda x:json.loads(x), taxi3))
            taxi4 = open(filename4,'r').readlines()
            taxi4 = list(map(lambda x:json.loads(x), taxi4))
            
            for i in range(len(taxi1)):
                temp1 = taxi1[i].copy()
                temp2 = taxi2[i].copy()
                temp3 = taxi3[i].copy()
                temp4 = taxi4[i].copy()
                
                driverID = np.unique([temp1['driverID'], temp2['driverID'],
                                      temp3['driverID'], temp4['driverID']])
                
                if len(driverID) <= 1:
                    temp1['speeds_history'] = temp2['speeds_history']
                    temp1['speeds_adjacent1'] = temp3['speeds_adjacent1']
                    temp1['speeds_adjacent2'] = temp3['speeds_adjacent2']
                    temp1['deepwalk_embedding'] = temp4['deepwalk_embedding']
                else:
                    print('date:',date,'   error:',i)
                
                json.dump(temp1, f, cls=MyEncoder)
                f.write('\n')
                
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi1))
            f.close()


def generate_iTCL_Train_Whole2():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_Whole2_byday, date_pool)


def generate_iTCL_Train_Whole3_byday(date):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH1 = DATA_PATH+'BeijingTaxi_iTCL_Train_Forward/'
    READ_DATA_PATH2 = DATA_PATH+'BeijingTaxi_iTCL_Train_History/'
    READ_DATA_PATH3 = DATA_PATH+'BeijingTaxi_iTCL_Train_Adjacent/'
    READ_DATA_PATH4 = DATA_PATH+'BeijingTaxi_iTCL_Train_SDNE/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Whole3/'
    
    filename1 = READ_DATA_PATH1+'2013-10-'+date[-2:]+'.json'
    filename2 = READ_DATA_PATH2+'2013-10-'+date[-2:]+'.json'
    filename3 = READ_DATA_PATH3+'2013-10-'+date[-2:]+'.json'
    filename4 = READ_DATA_PATH4+'2013-10-'+date[-2:]+'.json'
    
    print('Process:{}'.format(filename1))
    print('Process:{}'.format(filename2))
    print('Process:{}'.format(filename3))
    print('Process:{}'.format(filename4))
    
    if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3) and os.path.exists(filename4):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi1 = open(filename1,'r').readlines()
            taxi1 = list(map(lambda x:json.loads(x), taxi1))
            taxi2 = open(filename2,'r').readlines()
            taxi2 = list(map(lambda x:json.loads(x), taxi2))
            taxi3 = open(filename3,'r').readlines()
            taxi3 = list(map(lambda x:json.loads(x), taxi3))
            taxi4 = open(filename4,'r').readlines()
            taxi4 = list(map(lambda x:json.loads(x), taxi4))
            
            for i in range(len(taxi1)):
                temp1 = taxi1[i].copy()
                temp2 = taxi2[i].copy()
                temp3 = taxi3[i].copy()
                temp4 = taxi4[i].copy()
                
                driverID = np.unique([temp1['driverID'], temp2['driverID'],
                                      temp3['driverID'], temp4['driverID']])
                
                if len(driverID) <= 1:
                    temp1['speeds_history'] = temp2['speeds_history']
                    temp1['speeds_adjacent1'] = temp3['speeds_adjacent1']
                    temp1['speeds_adjacent2'] = temp3['speeds_adjacent2']
                    temp1['SDNE_embedding'] = temp4['SDNE_embedding']
                else:
                    print('date:',date,'   error:',i)
                
                json.dump(temp1, f, cls=MyEncoder)
                f.write('\n')
                
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi1))
            f.close()


def generate_iTCL_Train_Whole3():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_Whole3_byday, date_pool)


def generate_iTCL_Train_Whole_byday(date):
    DATA_PATH = '../../../dataset/Beijing/BeijingTaxi_iTCL/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Whole3/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Whole/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi = open(filename,'r').readlines()
            taxi = list(map(lambda x:json.loads(x), taxi))
            
            for i in range(len(taxi)):
                temp = taxi[i].copy()
                dist_gap = temp['dist_gap']
                grid_len1 = [0.0] + list(np.diff(dist_gap))
                grid_len2 = list(np.diff(dist_gap)) + [0.0]
                grid_len_list = list((np.asarray(grid_len1)+np.asarray(grid_len2))/2)
                temp['grid_len'] = grid_len_list
                
                json.dump(temp, f, cls=MyEncoder)
                f.write('\n')
                
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi))
            f.close()


def generate_iTCL_Train_Whole():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_Whole_byday, date_pool)


def generate_iTCL_Train_Test_byday(date):
    DATA_PATH = '../../../dataset/Beijing/BeijingTaxi_iTCL/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Whole/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Test/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            taxi = open(filename,'r').readlines()
            taxi = list(map(lambda x:json.loads(x), taxi))
            
            for i in range(1000):
                temp = taxi[i].copy()
                json.dump(temp, f, cls=MyEncoder)
                f.write('\n')
                
                if i%10000 == 0:
                    print('date:',date,'   process:',i/len(taxi))
            f.close()


def generate_iTCL_Train_Test():
    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_iTCL_Train_Test_byday, date_pool)


def calculate_statics():
    DATA_PATH = '../../../dataset/Beijing/BeijingTaxi_iTCL/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_iTCL_Train_Whole/'
    WRITE_DATA_PATH = DATA_PATH
    train_files = []
    test_files = ['2013-10-25.json','2013-10-26.json','2013-10-27.json','2013-10-28.json',
                  '2013-10-29.json','2013-10-30.json','2013-10-31.json']
    
    total_count = 0.0
#    road_count = 0.0
    tra_count = 0.0
    dist_gap_sum = 0.0
    dist_gap_sum_square = 0.0
    time_gap_sum = 0.0
    time_gap_sum_square = 0.0
    lngs_sum = 0.0
    lngs_sum_square = 0.0
    lats_sum = 0.0
    lats_sum_square = 0.0
    #dist_sum = 0.0
    dist_sum_square = 0.0
    #time_sum = 0.0
    time_sum_square = 0.0
    
    speeds_forward_sum = 0.0
    speeds_forward_square = 0.0
    speeds_adjacent1_sum = 0.0
    speeds_adjacent1_square = 0.0
    speeds_adjacent2_sum = 0.0
    speeds_adjacent2_square = 0.0
    speeds_history_sum = 0.0
    speeds_history_square = 0.0
    
#    road_embedding_sum = 0.0
#    road_embedding_square = 0.0
    
    grid_len_sum = 0.0
    grid_len_square = 0.0
    
    final_dict = {}
    
    date_pool = [str(20131000 + i) for i in range(8,25)]
    for date in date_pool:
        filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.json'
        print('Process:{}'.format(filename))
        if os.path.exists(filename):
            train_files.append('2013-10-'+date[-2:]+'.json')
            all_data = open(filename,'r').readlines()
            all_data = list(map(lambda x:json.loads(x), all_data))
            tra_count += len(all_data)
            for item in all_data:
                total_count += len(item['dist_gap']) - 1
#                road_count += len(item['road_embedding'])
                dist_gap_sum += item['dist']
                time_gap_sum += item['time']
                for i in range(1, len(item['dist_gap'])):
                    dist_gap_sum_square += ((item['dist_gap'][i] - item['dist_gap'][i-1])**2)
                    time_gap_sum_square += ((item['time_gap'][i] - item['time_gap'][i-1])**2)
                lngs_sum += sum(item['lngs'])
                lngs_sum_square += sum([i*i for i in item['lngs']])
                lats_sum += sum(item['lats'])
                lats_sum_square += sum([i*i for i in item['lats']])
    #            dist_sum += item['dist']
                dist_sum_square += item['dist']**2
    #            time_sum += item['time']
                time_sum_square += item['time']**2
                
                speeds_forward_sum += sum(item['speeds_forward'])
                speeds_forward_square += sum([i*i for i in item['speeds_forward']])
                speeds_adjacent1_sum += sum(item['speeds_adjacent1'])
                speeds_adjacent1_square += sum([i*i for i in item['speeds_adjacent1']])
                speeds_adjacent2_sum += sum(item['speeds_adjacent2'])
                speeds_adjacent2_square += sum([i*i for i in item['speeds_adjacent2']])
                speeds_history_sum += sum(item['speeds_history'])
                speeds_history_square += sum([i*i for i in item['speeds_history']])
                
#                road_embedding_sum += sum(item['road_embedding'])
#                road_embedding_square += sum([i*i for i in item['road_embedding']])
                
                grid_len_sum += sum(item['grid_len'])
                grid_len_square += sum([i*i for i in item['grid_len']])
    
    mean = dist_gap_sum / total_count
    mean_sq = dist_gap_sum_square / total_count
    final_dict['dist_gap_mean'] = mean
    final_dict['dist_gap_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = time_gap_sum / total_count
    mean_sq = time_gap_sum_square / total_count
    final_dict['time_gap_mean'] = mean
    final_dict['time_gap_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = lngs_sum / (total_count + tra_count)
    mean_sq = lngs_sum_square / (total_count + tra_count)
    final_dict['lngs_mean'] = mean
    final_dict['lngs_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = lats_sum / (total_count + tra_count)
    mean_sq = lats_sum_square / (total_count + tra_count)
    final_dict['lats_mean'] = mean
    final_dict['lats_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = dist_gap_sum / tra_count
    mean_sq = dist_sum_square / tra_count
    final_dict['dist_mean'] = mean
    final_dict['dist_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = time_gap_sum / tra_count
    mean_sq = time_sum_square / tra_count
    final_dict['time_mean'] = mean
    final_dict['time_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    
    mean = speeds_forward_sum / (total_count + tra_count) / 4
    mean_sq = speeds_forward_square / (total_count + tra_count) / 4
    final_dict['speeds_forward_mean'] = mean
    final_dict['speeds_forward_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = speeds_adjacent1_sum / (total_count + tra_count) / 4
    mean_sq = speeds_adjacent1_square / (total_count + tra_count) / 4
    final_dict['speeds_adjacent1_mean'] = mean
    final_dict['speeds_adjacent1_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = speeds_adjacent2_sum / (total_count + tra_count) / 4
    mean_sq = speeds_adjacent2_square / (total_count + tra_count) / 4
    final_dict['speeds_adjacent2_mean'] = mean
    final_dict['speeds_adjacent2_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = speeds_history_sum / (total_count + tra_count) / 7
    mean_sq = speeds_history_square / (total_count + tra_count) / 7
    final_dict['speeds_history_mean'] = mean
    final_dict['speeds_history_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    
#    mean = road_embedding_sum / road_count
#    mean_sq = road_embedding_square / road_count
#    final_dict['road_embedding_mean'] = mean
#    final_dict['road_embedding_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    
    mean = grid_len_sum / total_count
    mean_sq = grid_len_square / total_count
    final_dict['grid_len_mean'] = mean
    final_dict['grid_len_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))

    final_dict['train_set'] = train_files
    final_dict['eval_set'] = test_files
    final_dict['test_set'] = test_files
    f = open(WRITE_DATA_PATH+'config_iTCL2.json','w')
    json.dump(final_dict, f, indent=1)
    f.close()


if __name__ == '__main__':
#    generate_iTCL_Train_Forward()
#    generate_iTCL_Train_History()
#    generate_iTCL_Train_Adjacent()
#    generate_iTCL_Train_DeepWalk()
#    generate_iTCL_Train_SDNE()
#    generate_iTCL_Train_Whole1()
#    generate_iTCL_Train_Whole2()
#    generate_iTCL_Train_Whole3()
#    generate_iTCL_Train_Whole()
    generate_iTCL_Train_Test()
#    calculate_statics()
    print('test')