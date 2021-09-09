# Context-aware Entity Typing in Knowledge Graphs
This is the source code for: Context-aware Entity Typing in Knowledge Graphs.

### Requirements
+ Python 3
+ PyTorch >= 1.6.0
+ dgl >= 0.5.3

### Usage
Data preprocessing:
```
cd data
python preprocess.py --dataset FB15kET
python preprocess.py --dataset YAGO43kET
```
Train:
```
########### FB15kET ###########
# CET
python run.py --model CET --dataset FB15kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --temperature 0.5 --lr 0.001 --loss FNA --beta 4.0 --cuda

# R-GCN
python run.py --model RGCN --dataset FB15kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --lr 0.001 --loss FNA --beta 3.0 --cuda

# CompGCN
python run.py --model CompGCN --dataset FB15kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --lr 0.001 --loss FNA --activation relu --cuda

########### YAGO43kET ###########
# CET
python run.py --model CET --dataset YAGO43kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --temperature 0.5 --lr 0.001 --loss FNA --beta 2.0 --cuda

# R-GCN
python run.py --model RGCN --dataset YAGO43kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --lr 0.001 --loss FNA --beta 2.0 --cuda

# CompGCN
python run.py --model CompGCN --dataset YAGO43kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --lr 0.001 --loss FNA --activation relu --cuda
```
The KGE baselines can be found in KGE_baselines. 


### Acknowledgement
We refer to the code of <a href='https://github.com/dmlc/dgl'>DGL</a>. Thanks for their contributions.
