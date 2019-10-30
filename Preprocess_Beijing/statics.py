#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:50:22 2018

@author: shenyibin
"""

import os
import json
import datetime
import numpy as np
import pandas as pd

from preprocess import min_lng,max_lng,min_lat,max_lat
from preprocess import generate_date,geo_distance,generate_grid


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
    f = open(WRITE_DATA_PATH+'config_DeepTTE.json','w')
    json.dump(final_dict, f, indent=1)
    f.close()


if __name__ == '__main__':
    calculate_statics()
    print('test')