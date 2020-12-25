# TTPNet
Travel Time Prediction Based on Tensor Decomposition and Graph Embedding

Our paper: https://doi.org/10.1109/TKDE.2020.3038259

## Data Description
The file in data folder is a dataset sampled in Beijing in October 2013. Each file contains only 1,000 GPS-based trajectories, not the complete dataset.

## Model Training
python main.py --task train --result_file ./result/TTPNet.res  --log_file TTPNet

## Model Evaluation
python main.py --task test --weight_file ./saved_weights/weight --result_file ./result/TTPNet.res

