# KGE baselines
We provide the script to train and evaluate the KGE models here.

### Requirements
+ PyTorch >= 1.6.0
+ dglke == 0.1.2
+ dgl == 0.4.3

### Usage
```
########### FB15kET ###########
# TransE
# training
dglke_train --model_name TransE_l1 --data_path ../data/FB15kET/merge --dataset FB15kET \
--format raw_udd_hrt --data_files train.txt valid.txt test.txt \
--batch_size 1024 --log_interval 1000 \
--neg_sample_size 256 --hidden_dim 200 --gamma 6.5 --lr 0.023 --regularization_coef 7.20E-06 \
--batch_size_eval 16 --test -adv --gpu 0 --max_step 100000 -a 1.98

# evaluation
python eval.py --data_path ../data/FB15kET/merge/ --model_path ./ckpts/TransE_l1_FB15kET_0 \
--dataset FB15kET --model TransE_l1


# ComplEx
# training
dglke_train --model_name ComplEx --data_path ../data/FB15kET/merge --dataset FB15kET \
--format raw_udd_hrt --data_files train.txt valid.txt test.txt \
--batch_size 1024 --log_interval 1000 \
--neg_sample_size 512 --hidden_dim 200 --gamma 66.8 --lr 0.148 --regularization_coef 6.20E-06 \
--batch_size_eval 16 --test -adv --gpu 0 --max_step 100000 -a 2.00

# evaluation
python eval.py --data_path ../data/FB15kET/merge/ --model_path ./ckpts/ComplEx_FB15kET_0 \
--dataset FB15kET --model ComplEx


# RotatE
# training
dglke_train --model_name RotatE --data_path ../data/FB15kET/merge --dataset FB15kET \
--format raw_udd_hrt --data_files train.txt valid.txt test.txt \
--batch_size 1024 --log_interval 1000 \
--neg_sample_size 512 --regularization_coef 3.5e-06 --hidden_dim 200 --gamma 6.0 \
--lr 0.0168 --batch_size_eval 16 --test -adv -a 1.91 -de --max_step 100000 --gpu 0

# evaluation
python eval.py --data_path ../data/FB15kET/merge/ --model_path ./ckpts/RotatE_FB15kET_0 \
--dataset FB15kET --model RotatE --gamma 6.0


########### YAGO43kET ###########
# TransE
# training
dglke_train --model_name TransE_l1 --data_path ../data/YAGO43kET/merge --dataset YAGO43kET \
--format raw_udd_hrt --data_files train.txt valid.txt test.txt \
--batch_size 1024 --log_interval 1000 \
--neg_sample_size 512 --hidden_dim 200 --gamma 10.5 --lr 0.05 --regularization_coef 4.4E-06 \
--batch_size_eval 16 --test -adv --gpu 0 --max_step 100000 -a 1.99

# evaluation
python eval.py --data_path ../data/YAGO43kET/merge/ --model_path ./ckpts/TransE_l1_YAGO43kET_0 \
--dataset YAGO43kET --model TransE_l1


# ComplEx
# training
dglke_train --model_name ComplEx --data_path ../data/YAGO43kET/merge --dataset YAGO43kET \
--format raw_udd_hrt --data_files train.txt valid.txt test.txt \
--batch_size 1024 --log_interval 1000 \
--neg_sample_size 512 --hidden_dim 200 --gamma 62.4 --lr 0.154 --regularization_coef 2.00E-06 \
--batch_size_eval 16 --test -adv --gpu 0 --max_step 100000 -a 1.99

# evaluation
python eval.py --data_path ../data/YAGO43kET/merge/ --model_path ./ckpts/ComplEx_YAGO43kET_0 \
--dataset YAGO43kET --model ComplEx


# RotatE
# training
dglke_train --model_name RotatE --data_path ../data/YAGO43kET/merge --dataset YAGO43kET \
--format raw_udd_hrt --data_files train.txt valid.txt test.txt \
--batch_size 1024 --log_interval 1000 \
--neg_sample_size 256 --regularization_coef 2.4e-06 --hidden_dim 200 --gamma 11.8 \
--lr 0.0344 --batch_size_eval 16 --test -adv -a 1.74 -de --max_step 100000 --gpu 0

# evaluation
python eval.py --data_path ../data/YAGO43kET/merge/ --model_path ./ckpts/RotatE_YAGO43kET_0 \
--dataset YAGO43kET --model RotatE --gamma 11.8
```

You may need to change the '--model_path' according to your model save path. 
If you want to further adjust the hyperparameters, please ensure that the '--gamma' values used in training and evaluation are the same when evaluating RotatE.

### Acknowledgement
We refer to the code of <a href='https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding'>RotatE</a>. Thanks for their contributions.
