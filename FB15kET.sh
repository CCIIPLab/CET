# CET
python run.py --model CET --dataset FB15kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --temperature 0.5 --lr 0.001 --loss FNA --beta 4.0 --cuda

# R-GCN
python run.py --model RGCN --dataset FB15kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --lr 0.001 --loss FNA --beta 3.0 --cuda

# CompGCN
python run.py --model CompGCN --dataset FB15kET --load_ET --load_KG --neighbor_sampling \
--hidden_dim 100 --lr 0.001 --loss FNA --activation relu --cuda
