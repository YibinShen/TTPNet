#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:48:36 2019

@author: shenyibin
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats


def generate_roadnetwork_byday(date):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_RoadNetwork/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        
        roadnetwork_columns = ['current_grid', 'upstream_grid1']
        f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_RoadNetwork.txt', 'w')
        f.write(','.join(roadnetwork_columns)+'\n')
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
            temp_adjacent_grid1_df.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'_RoadNetwork.txt', index=0, header=0, mode='a')
                    
            if i%10000==0:
                print('date:',date,'   process:',i/len(trajectory_change))


def generate_roadnetwork():
#    date_pool = [str(20131000 + i) for i in range(1,25)]
    date_pool = [str(20131000 + i) for i in range(1,32)]
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(generate_roadnetwork_byday, date_pool)
#    for date in date_pool:
#        generate_roadnetwork_byday(date)

def generate_total_roadnetwork():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_RoadNetwork/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_RoadNetwork/'
    
    roadnetwork_df = pd.DataFrame()
#    date_pool = [str(20131000 + i) for i in range(1,25)]
    date_pool = [str(20131000 + i) for i in range(1,32)]
    for date in date_pool:
        filename1 = READ_DATA_PATH+'2013-10-'+date[-2:]+'_RoadNetwork.txt'
        print('Process:{}'.format(filename1))
        
        if os.path.exists(filename1):
            roadnetwork_date_df = pd.read_csv(filename1, header=0)
            roadnetwork_date_df = roadnetwork_date_df.drop_duplicates()
            
            roadnetwork_df = pd.concat([roadnetwork_df, roadnetwork_date_df])
            roadnetwork_df = roadnetwork_df.drop_duplicates()
        
    roadnetwork_df.to_csv(WRITE_DATA_PATH+'RoadNetwork.txt', index=0)
        
        

if __name__ == '__main__':
#    generate_roadnetwork()
    generate_total_roadnetwork()
    print('test')