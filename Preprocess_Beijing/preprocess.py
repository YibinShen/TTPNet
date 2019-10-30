# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:30:51 2018

@author: RUSSO
"""

import os
import time
import numpy as np
import pandas as pd

from math import radians, cos, sin, asin, sqrt
from coordTransform_utils import gcj02_to_wgs84

min_lng = 116.197918
max_lng = 116.548107
min_lat = 39.759773
max_lat = 40.025294


def generate_date():
    """
    Generate the date
    """
    date_pool = [str(20131000 + i) for i in range(1,32)]
    is_holiday = {}
    holiday_date = [str(20131000+i) for i in [1,2,3,4,5,6,7,12,13,19,20,26,27]]
    for date in date_pool:
        is_holiday[date] = 0
        if date in holiday_date:
            is_holiday[date] = 1
    day_of_week = {}
    for i in range(31):
        day_of_week[date_pool[i]] = (i+2)%7
    return date_pool,is_holiday,day_of_week


def geo_distance(lng1, lat1, lng2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lng1, lat1, lng2, lat2 = map(radians, map(float, [lng1, lat1, lng2, lat2]))
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371.393
    return c * r


def get_WGS84_lng(row):
    """
    transform the location to WGS84
    """
    longitude = row['lng']
    latitude = row['lat']
    new_lng,new_lat = gcj02_to_wgs84(longitude, latitude)
    return new_lng
def get_WGS84_lat(row):
    longitude = row['lng']
    latitude = row['lat']
    new_lng,new_lat = gcj02_to_wgs84(longitude, latitude)
    return new_lat


def generate_grid(min_lng, max_lng,
                  min_lat, max_lat,
                  grid_number):
    
    grid_lng_length = (max_lng-min_lng)/(grid_number-1)
    grid_lat_length = (max_lat-min_lat)/(grid_number-1)
    
    grid_dict = {'min_lng':min_lng, 'min_lat':min_lat,
                 'max_lng':max_lng, 'max_lat':max_lat,
                 'grid_lng_length':grid_lng_length, 'grid_lat_length':grid_lat_length}
    
    return grid_dict


def preprocess_taxi_after_byday(date):

    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_After/'
    
    date_time = '2013-10-'+date[-2:]
    for root, dirs, files in os.walk(READ_DATA_PATH+date_time):
#        print(root) #当前目录路径  
#        print(dirs) #当前路径下所有子目录  
#        print(files) #当前路径下所有非目录子文件
        
        taxi_write_columns = ['user_id', 'time', 'timestamp',
                              'lng', 'lat', 'speed', 'angle']
        
        if not os.path.exists(WRITE_DATA_PATH+date_time+'.txt'): 
            f = open(WRITE_DATA_PATH+date_time+'.txt', 'w')
            f.write(','.join(taxi_write_columns)+'\n')
            f.close()
            
            for file in files:
                filename = root+'/'+file
                print('Process:{}'.format(filename))
                
                taxi_columns = ['gps_time', 'user_id', 'lng', 'lat', 'speed', 
                                'angle', 'state1', 'state2', 'state3', 'receipt_time']
                taxi = pd.read_csv(filename, header=-1, names=taxi_columns)
                
                # state=1为有乘客
                taxi = taxi[taxi['state1']==1]
                # 经纬度删选
                taxi = taxi[(taxi['lng']>=min_lng) & (taxi['lng']<=max_lng) &
                            (taxi['lat']>=min_lat) & (taxi['lat']<=max_lat)]
                
                # 去除重复行
                taxi = taxi.drop_duplicates(['user_id', 'receipt_time'])
                print('filename:',filename, '   ', len(taxi))
                
                # 转换时间选择真实的日期和时间
                taxi['time'] = pd.to_datetime(taxi['receipt_time'], format='%Y%m%d%H%M%S')
                true_time = str.split(file, sep='.')[0]
                true_time = str.split(true_time, sep='-')[1]
                range_time = pd.date_range(start=date_time+' '+true_time+':00:00', periods=3, freq='h')
        
                taxi_after = taxi[(taxi['time']>=range_time[0]) & (taxi['time']<range_time[-1])].copy()
                taxi_after['timestamp'] = taxi_after['time'].apply(lambda x: int(time.mktime(x.timetuple())))
                
                taxi_write = taxi_after[taxi_write_columns]
                taxi_write = taxi_write.drop_duplicates(['user_id','time'])
                taxi_write.to_csv(WRITE_DATA_PATH+date_time+'.txt', index=0, header=0, mode='a')


def preprocess_taxi_after():

    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(15) as p:
        res = p.map(preprocess_taxi_after_byday, date_pool)
        return res
#    for date in date_pool:
#        preprocess_taxi_after_byday(date)


def split_trajectory_byday(date, time_interval=120, min_length=5, 
                           move_speed=0, road_speed=0.002):
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_After/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_SplitTrajectory/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt'):
            taxi = pd.read_csv(filename, header=0)
            taxi = taxi.drop_duplicates(['user_id', 'timestamp'])
            taxi.sort_values(['user_id','timestamp'], ascending=[True, True], inplace=True)
            taxi_res_columns = ['user_id','trajectory_id','timestamp',
                                'lng', 'lat', 'speed']
            
            f = open(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', 'w')
            f.write(','.join(taxi_res_columns)+'\n')
            f.close()
            
            user_id1 = taxi['user_id'].tolist()
            taxi['user_id1'] = [user_id1[0]] + user_id1[:-1]
            user_change = np.where(taxi['user_id']!=taxi['user_id1'])
            user_change = [0]+ list(user_change[0]) + [len(taxi)]
            
            for i in range(len(user_change)-1):
                temp = taxi[user_change[i]:user_change[i+1]].copy()
                temp = temp.sort_values(['timestamp'], ascending=[True])
                
                timestamp1 = temp['timestamp'].tolist()
                temp['timestamp1'] = [timestamp1[0]] + timestamp1[:-1]
                temp['dtimestamp'] = temp['timestamp'] - temp['timestamp1']
                temp = temp[(temp['dtimestamp']==0) | (temp['dtimestamp'] >= 3)]
                
                temp = temp.sort_values(['timestamp'], ascending=[True])
                
                lng1 = temp['lng'].tolist()
                lat1 = temp['lat'].tolist()
                timestamp1 = temp['timestamp'].tolist()
                temp['lng1'] = [lng1[0]] + lng1[:-1]
                temp['lat1'] = [lat1[0]] + lat1[:-1]
                temp['timestamp1'] = [timestamp1[0]] + timestamp1[:-1]
    
                temp['ddis'] = temp.apply(lambda x: geo_distance(
                        x['lng'], x['lat'], x['lng1'], x['lat1']), axis=1)
                temp['dtimestamp'] = temp['timestamp'] - temp['timestamp1']
                temp['road_speed'] = (temp['ddis']/temp['dtimestamp']).fillna(0)
    
                # 采样间隔大于120s认为是两条轨迹
                segment_loc = np.where((temp['dtimestamp']>=time_interval) | (temp['road_speed']>120/3600) | (temp['speed']>120))
                segment_loc = [0]+ list(segment_loc[0]) + [len(temp)]
                
                for j in range(len(segment_loc)-1):
                    new_temp = temp[segment_loc[j]:segment_loc[j+1]].copy()
                    if len(new_temp)>1 and new_temp['speed'].values[0]>120:
                        new_temp = new_temp[1:]
                    new_temp.drop_duplicates(['lng', 'lat'], keep='first', inplace=True)
                    if len(new_temp) >= min_length:
                        trajectory_id = str(temp['user_id'].values[0])+'_'+str(j+1)
                        new_temp['trajectory_id'] = trajectory_id
                        
                        for k in range(1,len(new_temp)-1):
                            if new_temp['speed'].values[k-1]>move_speed and new_temp['road_speed'].values[k]>road_speed:
                                break
                        for l in range(len(new_temp)-1, 0, -1):
                            if new_temp['speed'].values[l-1]>move_speed and new_temp['road_speed'].values[l]>road_speed:
                                break

                        taxi_res = new_temp[(k-1):(l+1)]
                        if len(taxi_res) >= min_length:
                            taxi_res = taxi_res[taxi_res_columns]
#                            taxi_res['ddis'].values[0] = 0.0
                            taxi_res.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0, header=0, mode='a')

                if i%1000 == 0:
                    print('date:',date,'   process:',i/len(user_change))


def split_trajectory():

    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(10) as p:
        p.map(split_trajectory_byday, date_pool)
#    for date in date_pool:
#        split_trajectory_byday(date)


def generate_area():
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_SplitTrajectory/'
    WRITE_DATA_PATH = DATA_PATH
    
    min_lng_area = 180
    max_lng_area = 0
    min_lat_area = 90
    max_lat_area = 0
    
    for root, dirs, files in os.walk(READ_DATA_PATH):
        for file in files:
            filename = root+file
            print('Process:{}'.format(filename))
            
            taxi = pd.read_csv(filename, header=0)
            
            min_lng_area = min(min_lng_area, min(taxi['lng']))
            max_lng_area = max(max_lng_area, max(taxi['lng']))
            min_lat_area = min(min_lat_area, min(taxi['lat']))
            max_lat_area = max(max_lat_area, max(taxi['lat']))
        
    area_df = pd.DataFrame({'min_lng_area':[min_lng_area], 'max_lng_area':[max_lng_area],
                            'min_lat_area':[min_lat_area], 'max_lat_area':[max_lat_area]})
    area_df.to_csv(WRITE_DATA_PATH+'Area_Beijing.txt', index=0)


def grid_trajectory_byday(date, grid_number=256):
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_SplitTrajectory/'
    WRITE_DATA_PATH = DATA_PATH+'BeijingTaxi_GridTrajectory/'
    
    filename = READ_DATA_PATH+'2013-10-'+date[-2:]+'.txt'
    print('Process:{}'.format(filename))
    
#    area = pd.read_csv(DATA_PATH+'Area_Beijing.txt', header=0)
#    area = area.iloc[0].to_dict()
    grid_dict = generate_grid(min_lng, max_lng,
                              min_lat, max_lat, grid_number)
    
    if os.path.exists(filename):
        if not os.path.exists(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt'):
            taxi_res_columns = ['user_id', 'trajectory_id', 'timestamp',
                                'lng', 'lat', 'grid_id', 'speed']
            taxi = pd.read_csv(filename, header=0)
            
            taxi['grid_lng_id'] = (taxi['lng']-grid_dict['min_lng']) // grid_dict['grid_lng_length']
            taxi['grid_lat_id'] = (taxi['lat']-grid_dict['min_lat']) // grid_dict['grid_lat_length']
            taxi['grid_id'] = taxi['grid_lng_id']*grid_number+taxi['grid_lat_id']
            taxi['grid_id'] = taxi['grid_id'].astype(int)
            
            if min(taxi['grid_id'])<0 or max(taxi['grid_id'])>256*256:
                print('date:',date,'Error!')
            
            taxi_res = taxi[taxi_res_columns]
            taxi_res.to_csv(WRITE_DATA_PATH+'2013-10-'+date[-2:]+'.txt', index=0)


def grid_trajectory():

    date_pool,is_holiday,day_of_week = generate_date()
    from multiprocessing import Pool

    with Pool(15) as p:
        p.map(grid_trajectory_byday, date_pool)
#    for date in date_pool:
#        grid_trajectory_byday(date)


if __name__ == '__main__':
#    preprocess_taxi_after()
#    split_trajectory()
#    generate_area()
#    grid_trajectory()
    print('test')