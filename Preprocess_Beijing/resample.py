#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:15:42 2018

@author: shenyibin
"""

import os
import json
import datetime
import numpy as np
import pandas as pd

from preprocess import min_lng,max_lng,min_lat,max_lat
from preprocess import generate_date,geo_distance,generate_grid


def get_userid_gridid():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridTrajectory/'
    WRITE_DATA_PATH = DATA_PATH
    
    user_id = set()
    grid_id = set()
    
    for root, dirs, files in os.walk(READ_DATA_PATH):    
        for file in files:
            filename = root+file
            print('Process:{}'.format(filename))
            
            taxi = pd.read_csv(filename, header=0)
            taxi['user_id'] = taxi['user_id'].apply(lambda x: str(x))
            taxi['grid_id'] = taxi['grid_id'].apply(lambda x: str(x))
            
            user_id_temp = set(taxi['user_id'])
            grid_id_temp = set(taxi['grid_id'])
#            user_id_temp = list(np.unique(taxi['user_id']))
            user_id = user_id | user_id_temp
            grid_id = grid_id | grid_id_temp
            
    user_id = list(user_id)
    grid_id = list(grid_id)
    user_id = list(np.unique(user_id))
    grid_id = list(np.unique(grid_id))
    user_id_dict = {user_id[i]:i for i in range(len(user_id))}
    grid_id_dict = {grid_id[i]:i for i in range(len(grid_id))}

    f = open(WRITE_DATA_PATH+'user_id'+'.json','w')
    json.dump(user_id_dict, f)
    f.close()
    f = open(WRITE_DATA_PATH+'grid_id'+'.json','w')
    json.dump(grid_id_dict, f)
    f.close()


def resampledata_byday(date, sample_dismin1=0.2, sample_dismin2=0.02,
                       min_length=5, min_time=120, max_time=7200,
                       min_dis=1, max_dis=50, min_speed=0.002):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridTrajectory/'
    WRITE_DATA_PATH1 = DATA_PATH+'BeijingTaxi_GridResample/'
    WRITE_DATA_PATH2 = DATA_PATH+'BeijingTaxi_GridResample2/'
    
    user_id_file = open(DATA_PATH + 'user_id.json', 'r')
    user_id_dict = json.loads(user_id_file.read())
    user_id_file.close()
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH1+'2013-10-'+date[-2:]+'.txt'):
            taxi_res_columns = ['driver_id','trajectory_id','timestamp',
                                'lng', 'lat', 'grid_id', 'time_gap', 'dis_gap', 'speed']
            
            taxi = pd.read_csv(filename, header=0)
            taxi.drop_duplicates(['lng', 'lat', 'trajectory_id'], keep='first', inplace=True)
            taxi['driver_id'] = taxi['user_id'].apply(lambda x: user_id_dict[str(x)])
            
            f = open(WRITE_DATA_PATH1+'2013-10-'+date[-2:]+'.txt', 'w')
            f.write(','.join(taxi_res_columns)+'\n')
            f.close()
            f = open(WRITE_DATA_PATH2+'2013-10-'+date[-2:]+'.txt', 'w')
            f.write(','.join(taxi_res_columns)+'\n')
            f.close()
    
            trajectory_id1 = taxi['trajectory_id'].tolist()
            taxi['trajectory_id1'] = [trajectory_id1[0]] + trajectory_id1[:-1]
            trajectory_change = np.where(taxi['trajectory_id']!=taxi['trajectory_id1'])
            trajectory_change = [0]+ list(trajectory_change[0]) + [len(taxi)]
            
            for i in range(len(trajectory_change)-1):
                temp = taxi[trajectory_change[i]:trajectory_change[i+1]].copy()
                
                lng1 = temp['lng'].tolist()
                lat1 = temp['lat'].tolist()
                temp['lng1'] = [lng1[0]] + lng1[:-1]
                temp['lat1'] = [lat1[0]] + lat1[:-1]
                
                temp['ddis'] = temp.apply(lambda x: geo_distance(
                    x['lng'], x['lat'], x['lng1'], x['lat1']), axis=1)
                
                temp['time_gap'] = temp['timestamp']-temp['timestamp'].values[0] + 0.0
                temp['dis_gap'] = 0.0
                temp['resample1_dis'] = 0.0
                temp['resample2_dis'] = 0.0
                
                for j in range(1,len(temp)):
                    temp['dis_gap'].values[j] = temp['dis_gap'].values[j-1]+temp['ddis'].values[j]
                    temp['resample1_dis'].values[j] = temp['resample1_dis'].values[j-1]+temp['ddis'].values[j]
                    temp['resample2_dis'].values[j] = temp['resample2_dis'].values[j-1]+temp['ddis'].values[j]
                    if temp['resample1_dis'].values[j] >= sample_dismin1:
                        temp['resample1_dis'].values[j] = 0.0
                    if temp['resample2_dis'].values[j] >= sample_dismin2:
                        temp['resample2_dis'].values[j] = 0.0
                # 最后一个采样点
                temp['resample1_dis'].values[-1] = 0.0
                temp['resample2_dis'].values[-1] = 0.0
                
                temp_resample1 = temp[temp['resample1_dis']==0.0]
                temp_resample2 = temp[temp['resample2_dis']==0.0]
                
#                if sum(np.diff(temp_resample2['time_gap'])>120)>2:
#                    print('i:',i,
#                          'sum:',sum(np.diff(temp_resample2['time_gap'])>120),
#                          'speed:', temp_resample1['dis_gap'].values[-1]/temp_resample1['time_gap'].values[-1])
                
                if len(temp_resample1)>=min_length and temp_resample1['time_gap'].values[-1]>=min_time and temp_resample1['time_gap'].values[-1]<=max_time and temp_resample1['dis_gap'].values[-1]>=min_dis and temp_resample1['dis_gap'].values[-1]<=max_dis and temp_resample1['dis_gap'].values[-1]/temp_resample1['time_gap'].values[-1]>min_speed:
#                    if sum(np.diff(temp_resample2['time_gap'])>120)==0:
                    temp_resample1 = temp_resample1[taxi_res_columns]
                    temp_resample2 = temp_resample2[taxi_res_columns]
                    temp_resample1.to_csv(WRITE_DATA_PATH1+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')
                    temp_resample2.to_csv(WRITE_DATA_PATH2+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')
                if i%10000 == 0:
                        print('date:',date,'   process:',i/len(trajectory_change))


def resampledata():
    
    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(10) as p:
        p.map(resampledata_byday, date_pool)
#    for date in date_pool:
#        generate_resampledata_byday(date)


def choosedata_byday(date, min_length=5, min_time=60, max_time=7200,
                     max_dis=50, min_speed=0.002):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample2_Before/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample2/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        taxi_res_columns = ['driver_id','trajectory_id','timestamp',
                            'WGS84_lng', 'WGS84_lat', 'grid_id', 'time_gap', 'dis_gap']
        f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', 'w')
        f.write(','.join(taxi_res_columns)+'\n')
        f.close()
        taxi = pd.read_csv(filename, header=0)
        
        trajectory_id1 = taxi['trajectory_id'].tolist()
        taxi['trajectory_id1'] = [trajectory_id1[0]] + trajectory_id1[:-1]
        trajectory_change = np.where(taxi['trajectory_id']!=taxi['trajectory_id1'])
        trajectory_change = [0]+ list(trajectory_change[0]) + [len(taxi)]
        
        for i in range(len(trajectory_change)-1):
            taxi_temp = taxi[trajectory_change[i]:trajectory_change[i+1]].copy()
            if len(taxi_temp)>=min_length and taxi_temp['time_gap'].values[-1]>=min_time and taxi_temp['time_gap'].values[-1]<=max_time and taxi_temp['dis_gap'].values[-1]<=max_dis and taxi_temp['dis_gap'].values[-1]/taxi_temp['time_gap'].values[-1]>min_speed:
                taxi_temp = taxi_temp[taxi_res_columns]
                taxi_temp.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')


def choosedata():
    
    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(choosedata_byday, date_pool)
#    for date in date_pool:
#        choosedata_byday(date)


def choosedata_final_byday(date, min_time=120):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample2/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_GridResample2_final/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        taxi_res_columns = ['driver_id','trajectory_id','timestamp',
                            'WGS84_lng', 'WGS84_lat', 'grid_id', 'time_gap', 'dis_gap']
        f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', 'w')
        f.write(','.join(taxi_res_columns)+'\n')
        f.close()
        taxi = pd.read_csv(filename, header=0)
        
        trajectory_id1 = taxi['trajectory_id'].tolist()
        taxi['trajectory_id1'] = [trajectory_id1[0]] + trajectory_id1[:-1]
        trajectory_change = np.where(taxi['trajectory_id']!=taxi['trajectory_id1'])
        trajectory_change = [0]+ list(trajectory_change[0]) + [len(taxi)]
        
        for i in range(len(trajectory_change)-1):
            taxi_temp = taxi[trajectory_change[i]:trajectory_change[i+1]].copy()
            if taxi_temp['time_gap'].values[-1]>=min_time:
                taxi_temp = taxi_temp[taxi_res_columns]
                taxi_temp.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')


def choosedata_final():
    
    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(5) as p:
        p.map(choosedata_final_byday, date_pool)
#    for date in date_pool:
#        choosedata_final_byday(date)


def calculate_statics():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Train1/'
    WRITE_DATA_PATH = DATA_PATH
    train_files = []
    test_files = ['2013-10-25.json','2013-10-26.json','2013-10-27.json','2013-10-28.json',
                  '2013-10-29.json','2013-10-30.json','2013-10-31.json']
    
    total_count = 0.0
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
                dist_gap_sum += item['dist']
                time_gap_sum += item['time']
                for i in range(1,len(item['dist_gap'])):
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
    
    mean = dist_gap_sum/total_count
    mean_sq = dist_gap_sum_square/total_count
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
    
    final_dict['train_set'] = train_files
    final_dict['eval_set'] = test_files
    final_dict['test_set'] = test_files
    f = open(WRITE_DATA_PATH+'config.json','w')
    json.dump(final_dict, f, indent=1)
    f.close()

def calculate_statics_speed_more():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_Train2/'
    WRITE_DATA_PATH = DATA_PATH
    train_files = []
    test_files = ['2013-10-25.json','2013-10-26.json','2013-10-27.json','2013-10-28.json',
                  '2013-10-29.json','2013-10-30.json','2013-10-31.json']
    
    total_count = 0.0
    tra_count = 0.0
    dist_gap_sum = 0.0
    dist_gap_sum_square = 0.0
    time_gap_sum = 0.0
    time_gap_sum_square = 0.0
    lngs_sum = 0.0
    lngs_sum_square = 0.0
    lats_sum = 0.0
    lats_sum_square = 0.0
    
    speeds_forward_60_sum = 0.0
    speeds_forward_60_square = 0.0
    speeds_forward_45_sum = 0.0
    speeds_forward_45_square = 0.0
    speeds_forward_30_sum = 0.0
    speeds_forward_30_square = 0.0
    speeds_forward_15_sum = 0.0
    speeds_forward_15_square = 0.0
    
    #dist_sum = 0.0
    dist_sum_square = 0.0
    #time_sum = 0.0
    time_sum_square = 0.0
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
                dist_gap_sum += item['dist']
                time_gap_sum += item['time']
                for i in range(1,len(item['dist_gap'])):
                    dist_gap_sum_square += ((item['dist_gap'][i] - item['dist_gap'][i-1])**2)
                    time_gap_sum_square += ((item['time_gap'][i] - item['time_gap'][i-1])**2)
                lngs_sum += sum(item['lngs'])
                lngs_sum_square += sum([i*i for i in item['lngs']])
                lats_sum += sum(item['lats'])
                lats_sum_square += sum([i*i for i in item['lats']])
                
                speeds_forward_60_sum += sum(item['speeds_forward_60'])
                speeds_forward_60_square += sum([i*i for i in item['speeds_forward_60']])
                speeds_forward_45_sum += sum(item['speeds_forward_45'])
                speeds_forward_45_square += sum([i*i for i in item['speeds_forward_45']])
                speeds_forward_30_sum += sum(item['speeds_forward_30'])
                speeds_forward_30_square += sum([i*i for i in item['speeds_forward_30']])
                speeds_forward_15_sum += sum(item['speeds_forward_15'])
                speeds_forward_15_square += sum([i*i for i in item['speeds_forward_15']])
                              
    #            dist_sum += item['dist']
                dist_sum_square += item['dist']**2
    #            time_sum += item['time']
                time_sum_square += item['time']**2
    
    mean = dist_gap_sum/total_count
    mean_sq = dist_gap_sum_square/total_count
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
    
    mean = speeds_forward_60_sum / (total_count + tra_count)
    mean_sq = speeds_forward_60_square / (total_count + tra_count)    
    final_dict['speeds_forward_60_mean'] = mean
    final_dict['speeds_forward_60_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = speeds_forward_45_sum / (total_count + tra_count)
    mean_sq = speeds_forward_45_square / (total_count + tra_count)    
    final_dict['speeds_forward_45_mean'] = mean
    final_dict['speeds_forward_45_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = speeds_forward_30_sum / (total_count + tra_count)
    mean_sq = speeds_forward_30_square / (total_count + tra_count)    
    final_dict['speeds_forward_30_mean'] = mean
    final_dict['speeds_forward_30_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = speeds_forward_15_sum / (total_count + tra_count)
    mean_sq = speeds_forward_15_square / (total_count + tra_count)    
    final_dict['speeds_forward_15_mean'] = mean
    final_dict['speeds_forward_15_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    
    mean = dist_gap_sum / tra_count
    mean_sq = dist_sum_square / tra_count
    final_dict['dist_mean'] = mean
    final_dict['dist_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    mean = time_gap_sum / tra_count
    mean_sq = time_sum_square / tra_count
    final_dict['time_mean'] = mean
    final_dict['time_std'] = np.sqrt(np.abs(mean_sq - np.square(mean)))
    
    final_dict['train_set'] = train_files
    final_dict['eval_set'] = test_files
    final_dict['test_set'] = test_files
    f = open(WRITE_DATA_PATH+'config.json','w')
    json.dump(final_dict, f, indent=1)
    f.close()

    
if __name__ == '__main__':
#    get_userid_gridid()
#    resampledata()
#    choosedata()
#    choosedata_final()
#    calculate_statics()
#    calculate_statics_speed_more()
    print('test')
