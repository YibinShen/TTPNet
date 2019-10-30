#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:50:04 2018

@author: shenyibin
"""

import os
import json
import datetime
import numpy as np
import pandas as pd

from preprocess import min_lng,max_lng,min_lat,max_lat
from preprocess import generate_date,geo_distance,generate_grid


def generate_traindata1_byday(date, min_length=5):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_Train1/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json'):
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.json', 'w')
            _,is_holiday,day_of_week = generate_date()
            taxi = pd.read_csv(filename, header=0)
            
            trajectory_id1 = taxi['trajectory_id'].tolist()
            taxi['trajectory_id1'] = [trajectory_id1[0]] + trajectory_id1[:-1]
            trajectory_change = np.where(taxi['trajectory_id']!=taxi['trajectory_id1'])
            trajectory_change = [0]+ list(trajectory_change[0]) + [len(taxi)]
            basetime = int(datetime.datetime.timestamp(datetime.datetime(2013,10,int(date[-2:]))))

            for i in range(len(trajectory_change)-1):
                taxi_temp = taxi[trajectory_change[i]:trajectory_change[i+1]].copy()
                if len(taxi_temp)>= min_length:
                    time_gap_list = list(taxi_temp['time_gap'])
                    total_dist = taxi_temp['dis_gap'].values[-1]
                    lat_list = list(taxi_temp['lat'])
                    lng_list = list(taxi_temp['lng'])
                    driver_id = int(taxi_temp['driver_id'].values[0])
                    week_id = day_of_week[date]
                    time_id = int((taxi_temp['timestamp'].values[0]-basetime) // 60)
                    # 节假日
                    date_id = is_holiday[date]
                    time_interval = taxi_temp['time_gap'].values[-1]
                    dist_gap_list = list(taxi_temp['dis_gap'])
                    grid_id_list = list(taxi_temp['grid_id'])
                    
                    taxi_res = {
                    'time_gap': time_gap_list,
                    'dist': total_dist,
                    'lats': lat_list,
                    'lngs': lng_list,
                    'driverID': driver_id,
                    'weekID': week_id,
                    'timeID': time_id,
                    'dateID': date_id,
                    'time': time_interval,
                    'dist_gap': dist_gap_list,
                    'grid_id': grid_id_list
                    }
                    json.dump(taxi_res, f)
                    f.write('\n')
            f.close()


def generate_traindata1():

    date_pool = [str(20131000 + i) for i in range(8,32)]
    from multiprocessing import Pool

    with Pool(10) as p:
        p.map(generate_traindata1_byday, date_pool)
#    for date in date_pool:
#        generate_traindata1_byday(date)


if __name__ == '__main__':
#    generate_traindata1()
    print('test')