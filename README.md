# CDGP
The code is for our paper [Community-based Dynamic Graph Learning for Popularity Prediction](https://dl.acm.org/doi/abs/10.1145/3580305.3599281).

## Train
```python

python train.py --gpu 0 --prefix aminer --lr 0.00001 --patience 20 -d aminer 

### End2End

python train.py --gpu 0 --prefix aminer --lr 0.00001 --patience 20 -d aminer --end2end

### Pre-train with edge prediction task

python pretrain.py --gpu 0 --prefix aminer --lr 0.0001 --patience 20 -d aminer
