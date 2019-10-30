## Model Training
python main.py --task train  --batch_size 256  --result_file ./result/TTPNet.res  --log_file TTPNet

## Model Evaluation
python main.py --task test --weight_file ./saved_weights/weight --batch_size 256  --result_file ./result/TTPNet.res