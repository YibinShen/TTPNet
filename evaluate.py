#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:12:20 2018

@author: shenyibin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'result256/'
result_list = ['TTPNet_256_test.res', 'TTPNet2_256_test.res',
               'TTPNet3_256_test.res', 'TTPNet6_256_test.res',
               'TTPNet5_256_test.res']
res_columns = ['weekID', 'timeID', 'driverID', 'dist', 'label', 'pred']
res1 = pd.read_csv(DATA_PATH+result_list[4], header=-1, names=res_columns)

res1['hourID'] = res1['timeID'] // 60
res1['MAPE'] = abs(res1['pred']-res1['label'])/res1['label']
res1['MAE'] = abs(res1['pred']-res1['label'])
res1['RMSE'] = (res1['pred']-res1['label'])**2
    
print('MAE:', np.mean(res1['MAE']))
print('MAPE:', np.mean(res1['MAPE'])*100)
print('RMSE:', np.sqrt(np.mean(res1['RMSE'])))
print('SR:', sum(res1['MAPE']<=0.1)/len(res1)*100)

res1['distID'] = res1['dist'] // 5
res1['distID'] = res1['distID'].apply(lambda x: 6 if x>6 else x)
res1_dist = res1[['distID', 'MAPE']].groupby(['distID'], as_index=False).mean()
#res1_dist.to_csv('dist.csv', index=0)
#plt.plot(res1_dist['MAPE'])

res1_weekday = res1[(res1['weekID']!=0) & (res1['weekID']!=6)]
res1_weekend = res1[(res1['weekID']==0) | (res1['weekID']==6)]

res1_weekday_hour = res1_weekday[['hourID', 'MAPE']].groupby(['hourID'], as_index=False).mean()
#plt.plot(res1_weekday_hour['MAPE'])
res1_weekend_hour = res1_weekend[['hourID', 'MAPE']].groupby(['hourID'], as_index=False).mean()
#plt.plot(res1_weekend_hour['MAPE'])
res1_hour = res1[['hourID', 'MAPE']].groupby(['hourID'], as_index=False).mean()
#plt.plot(res1_hour['MAPE'])
res1_weekday_hour = res1_weekday_hour['MAPE']*100
res1_weekend_hour = res1_weekend_hour['MAPE']*100

a=res1[((res1['hourID']>=7) & (res1['hourID']<=9)) | ((res1['hourID']>=16) & (res1['hourID']<=18))]
print('MAPE:', np.mean(a['MAPE'])*100)
print('SR:', sum(a['MAPE']<=0.1)/len(a)*100)

