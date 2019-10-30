#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:25:45 2018

@author: shenyibin
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats


def generate_total_history():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_MatrixFactorization/'
    WRITE_DATA_PATH = DATA_PATH
    
    tensor_res_columns = ['grid_id', 'time_id', 'date_id', 'speed', 'grid_speed']
    f = open(WRITE_DATA_PATH+'Tensor_total_history.txt', 'w')
    f.write(','.join(tensor_res_columns)+'\n')
    f.close()
    
    date_pool = [str(20131000 + i) for i in range(1,25)]
    for i in range(len(date_pool)):
        filename = READ_DATA_PATH+'2013-10-'+date_pool[i][-2:]+'.txt'
        print('Process:{}'.format(filename))
        if os.path.exists(filename):
            taxi_history = pd.read_csv(filename, header=0)
            taxi_history['date_id'] = i
            taxi_history = taxi_history[tensor_res_columns]
            taxi_history.to_csv(WRITE_DATA_PATH+'Tensor_total_history.txt', index=0, header=0, mode='a')


def generate_speed_distribution():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH
    WRITE_DATA_PATH = DATA_PATH
    
    filename = READ_DATA_PATH+'Tensor_total_history.txt'
    print('Process:{}'.format(filename))
    if os.path.exists(filename):
        
        taxi_history = pd.read_csv(filename, header=0)
        taxi_speed = taxi_history[['time_id','grid_id','speed']].groupby(['time_id','grid_id'],as_index=False).mean()
        taxi_speed.sort_values(['grid_id'], ascending=True, inplace=True)
        
        grid_id1 = taxi_speed['grid_id'].tolist()
        taxi_speed['grid_id1'] = [grid_id1[0]] + grid_id1[:-1]
        grid_change = np.where(taxi_speed['grid_id']!=taxi_speed['grid_id1'])
        grid_change = [0] + list(grid_change[0]) + [len(taxi_speed)]
        
        speed_df = pd.DataFrame()
        for i in range(len(grid_change)-1):
            temp = taxi_speed[grid_change[i]:grid_change[i+1]].copy()
            if len(temp) >= 70:
                temp_speed = np.zeros(96)
                for i in range(len(temp)):
                    temp_speed[temp['time_id'].values[i]] = temp['speed'].values[i]
                temp_speed_df = pd.DataFrame([temp_speed])
                temp_speed_df['grid_id'] = temp['grid_id'].values[0]
                speed_df = pd.concat([speed_df, temp_speed_df])
            
            if i%10000==0:
                print('process:',i/len(grid_change))
        
        speed_df.to_csv(WRITE_DATA_PATH+'Grid_speed_distribution.txt', index=0)


def generate_adjacent_grid_byday(date):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample2/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        
        adjacent_grid1_columns = ['current_grid', 'upstream_grid1']
        f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_Adjacent_grid1.txt', 'w')
        f.write(','.join(adjacent_grid1_columns)+'\n')
        f.close()
        adjacent_grid2_columns = ['current_grid', 'upstream_grid2']
        f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_Adjacent_grid2.txt', 'w')
        f.write(','.join(adjacent_grid2_columns)+'\n')
        f.close()
        
        taxi = pd.read_csv(filename, header=0)
        trajectory_id1 = taxi['trajectory_id'].tolist()
        taxi['trajectory_id1'] = [trajectory_id1[0]] + trajectory_id1[:-1]
        trajectory_change = np.where(taxi['trajectory_id']!=taxi['trajectory_id1'])
        trajectory_change = [0]+ list(trajectory_change[0]) + [len(taxi)]
        
        for i in range(len(trajectory_change)-1):
            temp = taxi[trajectory_change[i]:trajectory_change[i+1]].copy()
            grid_id1 = temp['grid_id'].tolist()
            temp_adjacent_grid1_df = pd.DataFrame({'current_grid':grid_id1})
            temp_adjacent_grid1_df['upstream_grid1'] = [grid_id1[0]] + grid_id1[:-1]
            temp_adjacent_grid1_df = temp_adjacent_grid1_df[temp_adjacent_grid1_df['current_grid']!=temp_adjacent_grid1_df['upstream_grid1']]
            temp_adjacent_grid1_df.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_Adjacent_grid1.txt', index=0, header=0, mode='a')
        
            grid_id2 = temp_adjacent_grid1_df['upstream_grid1'].tolist()
            if len(grid_id2)>1:
                temp_adjacent_grid2_df = pd.DataFrame({'current_grid':temp_adjacent_grid1_df['current_grid'].tolist()})
                temp_adjacent_grid2_df['upstream_grid2'] = [grid_id2[0]] + grid_id2[:-1]
                temp_adjacent_grid2_df = temp_adjacent_grid2_df[temp_adjacent_grid2_df['current_grid']!=temp_adjacent_grid2_df['upstream_grid2']]
                temp_adjacent_grid2_df = temp_adjacent_grid2_df.drop(temp_adjacent_grid2_df.index[0])
                temp_adjacent_grid2_df.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_Adjacent_grid2.txt', index=0, header=0, mode='a')
            
            if i%10000==0:
                print('date:',date,'   process:',i/len(trajectory_change))


def generate_adjacent_grid():
#    date_pool = [str(20131000 + i) for i in range(1,25)]
    date_pool = [str(20131000 + i) for i in range(1,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_adjacent_grid_byday, date_pool)
#    for date in date_pool:
#        generate_adjacent_grid_byday(date)

def generate_total_adjacent_grid():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    
    adjacent_grid1_df = pd.DataFrame()
    adjacent_grid2_df = pd.DataFrame()
#    date_pool = [str(20131000 + i) for i in range(1,25)]
    date_pool = [str(20131000 + i) for i in range(1,32)]
    for date in date_pool:
        filename1 = READ_DATA_PATH+'2013-10-'+date[-2:]+'_Adjacent_grid1.txt'
        filename2 = READ_DATA_PATH+'2013-10-'+date[-2:]+'_Adjacent_grid2.txt'
        print('Process:{}'.format(filename1))
        print('Process:{}'.format(filename2))
        
        if os.path.exists(filename1):
            adjacent_grid1_date_df = pd.read_csv(filename1, header=0)
            adjacent_grid2_date_df = pd.read_csv(filename2, header=0)
            adjacent_grid1_date_df = adjacent_grid1_date_df.drop_duplicates()
            adjacent_grid2_date_df = adjacent_grid2_date_df.drop_duplicates()
            
            adjacent_grid1_df = pd.concat([adjacent_grid1_df, adjacent_grid1_date_df])
            adjacent_grid2_df = pd.concat([adjacent_grid2_df, adjacent_grid2_date_df])
            adjacent_grid1_df = adjacent_grid1_df.drop_duplicates()
            adjacent_grid2_df = adjacent_grid2_df.drop_duplicates()
        
    adjacent_grid1_df.to_csv(WRITE_DATA_PATH+'Adjacent_grid1.txt', index=0)
    adjacent_grid2_df.to_csv(WRITE_DATA_PATH+'Adjacent_grid2.txt', index=0)
        
        
def calculate_adjacent_grid_similar1():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    
    speed_distribution = pd.read_csv(DATA_PATH+'Grid_speed_distribution.txt')
    adjacent_grid1_df = pd.read_csv(READ_DATA_PATH+'Adjacent_grid1.txt', header=0)
    adjacent_grid1_df['KL_current'] = -1
    adjacent_grid1_df['KL_forward_15'] = -1
    adjacent_grid1_df['KL'] = -1
#    adjacent_grid2_df = pd.read_csv(READ_DATA_PATH+'Adjacent_grid2.txt', header=0)
    
    for i in range(len(adjacent_grid1_df)):
        current_grid = speed_distribution[speed_distribution['grid_id']==adjacent_grid1_df['current_grid'].values[i]]
        upstream_grid1 = speed_distribution[speed_distribution['grid_id']==adjacent_grid1_df['upstream_grid1'].values[i]]
        if len(current_grid)>0 and len(upstream_grid1)>0:
            current_grid = current_grid.values[0]
            upstream_grid1 = upstream_grid1.values[0]
            current_grid = current_grid[:-1]
            upstream_grid1 = upstream_grid1[:-1]
            
            x = current_grid[(current_grid!=0) & (upstream_grid1!=0)]
            y = upstream_grid1[(current_grid!=0) & (upstream_grid1!=0)]
            
            adjacent_grid1_df['KL_current'].values[i] = scipy.stats.entropy(x, y)
            # 当前延后15分钟
            x = x[1:]
            y = y[:-1]
            adjacent_grid1_df['KL_forward_15'].values[i] = scipy.stats.entropy(x, y)
            
            adjacent_grid1_df['KL'].values[i] = min(adjacent_grid1_df['KL_current'].values[i],
                                                    adjacent_grid1_df['KL_forward_15'].values[i])
        if i%100000 == 0:
            print('process:',i/len(adjacent_grid1_df))
    adjacent_grid1_res = adjacent_grid1_df[adjacent_grid1_df['KL']!=-1]
    adjacent_grid1_res.to_csv(WRITE_DATA_PATH+'Adjacent_grid1_KL.txt', index=0)


def calculate_adjacent_grid_similar2():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    
    speed_distribution = pd.read_csv(DATA_PATH+'Grid_speed_distribution.txt')
    adjacent_grid2_df = pd.read_csv(READ_DATA_PATH+'Adjacent_grid2.txt', header=0)
    adjacent_grid2_df['KL_forward_15'] = -1
    adjacent_grid2_df['KL_forward_30'] = -1
    adjacent_grid2_df['KL'] = -1
    
    for i in range(len(adjacent_grid2_df)):
        current_grid = speed_distribution[speed_distribution['grid_id']==adjacent_grid2_df['current_grid'].values[i]]
        upstream_grid2 = speed_distribution[speed_distribution['grid_id']==adjacent_grid2_df['upstream_grid2'].values[i]]
        if len(current_grid)>0 and len(upstream_grid2)>0:
            current_grid = current_grid.values[0]
            upstream_grid2 = upstream_grid2.values[0]
            current_grid = current_grid[:-1]
            upstream_grid2 = upstream_grid2[:-1]
            
            x = current_grid[(current_grid!=0) & (upstream_grid2!=0)]
            y = upstream_grid2[(current_grid!=0) & (upstream_grid2!=0)]
            # 当前延后15分钟
            x = x[1:]
            y = y[:-1]
            
            adjacent_grid2_df['KL_forward_15'].values[i] = scipy.stats.entropy(x, y)
            # 当前延后30分钟
            x = x[1:]
            y = y[:-1]
            adjacent_grid2_df['KL_forward_30'].values[i] = scipy.stats.entropy(x, y)
            
            adjacent_grid2_df['KL'].values[i] = min(adjacent_grid2_df['KL_forward_15'].values[i],
                                                    adjacent_grid2_df['KL_forward_30'].values[i])
        if i%100000 == 0:
            print('process:',i/len(adjacent_grid2_df))
    adjacent_grid2_res = adjacent_grid2_df[adjacent_grid2_df['KL']!=-1]
    adjacent_grid2_res.to_csv(WRITE_DATA_PATH+'Adjacent_grid2_KL.txt', index=0)


def generate_adjacent_list():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_AdjacentGrid/'
    WRITE_DATA_PATH = DATA_PATH
    
    adjacent_grid1_df = pd.read_csv(READ_DATA_PATH+'Adjacent_grid1_KL.txt')
    adjacent_grid2_df = pd.read_csv(READ_DATA_PATH+'Adjacent_grid2_KL.txt')
    
    adjacent_grid_columns = ['current_grid','upstream_grid','KL']
    adjacent_grid1 = adjacent_grid1_df[['current_grid','upstream_grid1','KL']]
    adjacent_grid2 = adjacent_grid2_df[['current_grid','upstream_grid2','KL']]
    adjacent_grid1.columns = adjacent_grid_columns
    adjacent_grid2.columns = adjacent_grid_columns
    adjacent_grid = pd.concat([adjacent_grid1,adjacent_grid2])
    adjacent_grid.drop_duplicates(['current_grid', 'upstream_grid'], inplace=True)
    adjacent_grid.sort_values(['current_grid','KL'], ascending=[True,True], inplace=True)
    
    current_grid1 = adjacent_grid['current_grid'].tolist()
    adjacent_grid['current_grid1'] = [current_grid1[0]] + current_grid1[:-1]
    current_grid_change = np.where(adjacent_grid['current_grid']!=adjacent_grid['current_grid1'])
    current_grid_change = [0]+ list(current_grid_change[0]) + [len(adjacent_grid)]
    
    adjacent_grid_dict1 = {}
    adjacent_grid_dict2 = {}
    
    for i in range(len(current_grid_change)-1):
        temp = adjacent_grid[current_grid_change[i]:current_grid_change[i+1]].copy()
        if len(temp) < 2:
            adjacent_grid_dict1.update({str(temp['current_grid'].values[0]): int(temp['upstream_grid'].values[0])})
            print('i:',i,'   less 2')
        else:
            adjacent_grid_dict1.update({str(temp['current_grid'].values[0]): int(temp['upstream_grid'].values[0])})
            adjacent_grid_dict2.update({str(temp['current_grid'].values[0]): int(temp['upstream_grid'].values[1])})

    f = open(WRITE_DATA_PATH+'adjacent_grid1'+'.json','w')
    json.dump(adjacent_grid_dict1, f)
    f.close()
    f = open(WRITE_DATA_PATH+'adjacent_grid2'+'.json','w')
    json.dump(adjacent_grid_dict2, f)
    f.close()


if __name__ == '__main__':
#    generate_total_history()
#    generate_speed_distribution()
    generate_adjacent_grid()
    generate_total_adjacent_grid()
#    calculate_adjacent_grid_similar1()
#    calculate_adjacent_grid_similar2()
#    generate_adjacent_list()
    print('test')