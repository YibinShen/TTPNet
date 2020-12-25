# TTPNet
Travel Time Prediction Based on Tensor Decomposition and Graph Embedding

Our paper: https://ieeexplore.ieee.org/document/9261122

## Model Training
python main.py --task train --result_file ./result/TTPNet.res  --log_file TTPNet

## Model Evaluation
python main.py --task test --weight_file ./saved_weights/weight --result_file ./result/TTPNet.res
