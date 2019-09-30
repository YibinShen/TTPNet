#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:12:20 2018

@author: shenyibin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'result512/'
result_list = ['TTPNet_512_test.res', 'TTPNet2_512_test.res',
               'TTPNet3_512_test.res', 'TTPNet4_512_test.res',
               'TTPNet5_512_test.res']
res_columns = ['weekID', 'timeID', 'driverID', 'dist', 'label', 'pred']
res1 = pd.read_csv(DATA_PATH+result_list[4], header=-1, names=res_columns)

res1['hourID'] = res1['timeID'] // 60
res1['MAPE'] = abs(res1['pred']-res1['label'])/res1['label']
res1['MAE'] = abs(res1['pred']-res1['label'])
res1['RMSE'] = (res1['pred']-res1['label'])**2
    
print('MAPE:', np.mean(res1['MAPE'])*100)
print('MAE:', np.mean(res1['MAE']))
print('RMSE:', np.sqrt(np.mean(res1['RMSE'])))

res1['distID'] = res1['dist'] // 5
res1['distID'] = res1['distID'].apply(lambda x: 6 if x>6 else x)
res1_dist = res1[['distID', 'MAPE']].groupby(['distID'], as_index=False).mean()
#res1_dist.to_csv('dist.csv', index=0)
plt.plot(res1_dist['MAPE'])

#res1['dist_id'] = res1['dist'] // 2
#
#res1_everyday_hour = res1[['weekID', 'hourID', 'MAPE']].groupby(['weekID','hourID'],as_index=False).mean()
#res1_totalday_hour = res1[['hourID', 'MAPE']].groupby(['hourID'],as_index=False).mean()
#res1_day = res1[['weekID', 'MAPE']].groupby(['weekID'],as_index=False).mean()
#res1_dist = res1[['dist_id', 'dist', 'MAPE']].groupby(['dist_id'], as_index=False).mean()

#plt.plot(res1_totalday_hour['hourID'],res1_totalday_hour['MAPE'])
#plt.show()

#plt.plot(res1_dist['dist_id'],res1_dist['MAPE'])
#plt.plot(res2_dist['dist_id'],res2_dist['MAPE'])
#plt.show()
