#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:37:21 2019

@author: shenyibin
"""

import os
import json
import numpy as np
import pandas as pd


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


def generate_edgelist_mapping_deepwalk():
    
    DATA_PATH = '../../../dataset/Beijing/'
    READ_DATA_PATH = DATA_PATH+'BeijingTaxi_RoadNetwork/'
    
    filename = READ_DATA_PATH + 'RoadNetwork.txt'
    
    if os.path.exists(filename):
        graph = pd.read_csv(filename, header=0)
        graph = graph.drop_duplicates()
        graph = graph[['upstream_grid1', 'current_grid']]
        
        graph['upstream_grid1'] = graph['upstream_grid1'].apply(lambda x: str(x))
        graph['current_grid'] = graph['current_grid'].apply(lambda x: str(x))
        
        grid_id = set(graph['upstream_grid1']) | set(graph['current_grid'])
        grid_id = list(grid_id)
        grid_id = list(np.unique(grid_id))
    
        grid_id_dict = {grid_id[i]:i for i in range(len(grid_id))}

        f = open(DATA_PATH+'grid_id_mapping'+'.json','w')
        json.dump(grid_id_dict, f, cls=MyEncoder)
        f.close()
        
        graph['upstream_grid1'] = graph['upstream_grid1'].apply(lambda x: grid_id_dict[str(x)])
        graph['current_grid'] = graph['current_grid'].apply(lambda x: grid_id_dict[str(x)])
        
        graph.sort_values(['upstream_grid1', 'current_grid'], ascending=[True, True], inplace=True)
        
#        f = open(DATA_PATH+'Beijing_edgelist_deepwalk.txt', 'w')
#        f.write(' '.join([str(max(max(graph['upstream_grid1']), max(graph['current_grid']))+1), 
#                          str(len(graph))])+'\n')
#        f.close()
        
        graph.to_csv(DATA_PATH+'Beijing_edgelist_deepwalk.txt', header=0, index=0, sep=' ')


def generate_edgelist_mapping_SDNE():
    
    DATA_PATH = '../../../dataset/Beijing/'
    filename = DATA_PATH+'Beijing_edgelist_deepwalk.txt'
    
    if os.path.exists(filename):
        graph = pd.read_csv(filename, header=-1, sep=' ')
        
        f = open(DATA_PATH+'Beijing_edgelist_SDNE.txt', 'w')
        f.write(' '.join([str(max(max(graph.iloc[:,0]), max(graph.iloc[:,1]))+1), 
                          str(len(graph))])+'\n')
        f.close()
        
        graph.to_csv(DATA_PATH+'Beijing_edgelist_SDNE.txt', header=0, index=0, sep=' ', mode='a')
        

if __name__ == '__main__':
#    generate_edgelist_mapping_deepwalk()
    generate_edgelist_mapping_SDNE()
    print('test')