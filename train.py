import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import os
import inspect
from torchvision import models
import pandas as pd

from evaluation.evaluation import eval_popularity_prediction
from model.CDGP import CDGP
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder,plot_train
from utils.data_processing import get_data_popularity, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=2, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--use_time', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--use_community', action='store_true',
                    help='Whether to augment the model with a node community')
parser.add_argument('--community_updater', type=str, default="topk", choices=[
  "topk", "kmeans"], help='Type of memory updater')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--community_num', type=int, default=800, help='community_num')
parser.add_argument('--community_score', type=str, default="leconv", help='Type of community'
                                                                        'score')
parser.add_argument('--member_num', type=int, default=200, help='member_num')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
USE_TIME = args.use_time
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
USE_COMMUNITY = args.use_community
COMMUNITY_UPDATER_TYPE = args.community_updater
COMMUNITY_NUM = args.community_num
MEMBER_NUM = args.member_num

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
full_data, train_data, val_data, test_data,node_features, edge_features ,n_nodes,n_edges = get_data_popularity(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

frame = inspect.currentframe()

update_time_1 = ['2003-06-30','2003-12-30','2004-06-30','2004-12-30','2005-06-30','2005-12-30',
               '2006-06-30','2006-12-30','2007-06-30','2007-12-30','2008-06-30','2008-12-30'
               ,'2009-06-30',]
update_time = ['2004-12-30','2006-12-30','2008-12-30']
update_time_int = pd.to_datetime(update_time).view(np.int64) // 10 ** 9

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  community_path = "results/{}_{}_com.pkl".format(args.prefix, i) if i > 0 else "results/{}_com.pkl".format(args.prefix)
  memory_path = "results/{}_{}_memory.pkl".format(args.prefix, i) if i > 0 else "results/{}_memory.pkl".format(args.prefix)
  dynamic_memory_path = "results/{}_{}_dynamic_memory.pkl".format(args.prefix, i) if i > 0 else "results/{}_dynamic_memory.pkl".format(args.prefix)
  dynamic_community_path = "results/{}_{}_dynamic_community.pkl".format(args.prefix, i) if i > 0 else "results/{}_dynamic_community.pkl".format(args.prefix)
  exp_path = "results/{}_{}_exp.pkl".format(args.prefix, i) if i > 0 else "results/{}_exp.pkl".format(args.prefix)

  results_write = open(results_path,'wb')
  community_write = open(community_path,'wb')
  memory_write = open(memory_path,'wb')
  dynamic_community_write=open(dynamic_community_path,'wb')
  dynamic_memory_write = open(dynamic_memory_path,'wb')
  exp_write=open(exp_path,'wb')

  Path("results/").mkdir(parents=True, exist_ok=True)


  # Initialize Model
  cdgp = CDGP(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features,
              device=device,
              n_nodes=n_nodes,
              n_edges=n_edges,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              use_time = USE_TIME,
              use_community=USE_COMMUNITY,
              community_updater_type = COMMUNITY_UPDATER_TYPE,
              community_num = COMMUNITY_NUM,
              community_score_type = args.community_score,
              member_num = MEMBER_NUM,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep)
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(cdgp.parameters(), lr=LEARNING_RATE)
  cdgp = cdgp.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  #new_nodes_val_aps = []
  #val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []
  train_loss_list = []
  val_loss_list= []
  val_rmsle_list = []
  val_msle_list = []
  val_pcc_list = []
  val_male_list = []
  val_mape_list = []

  batch_community_index_list = []




  early_stopper = EarlyStopMonitor(max_round=args.patience,higher_better=False)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    if USE_MEMORY:
      cdgp.memory.__init_memory__()

    cdgp.set_neighbor_finder(train_ngh_finder)
    logger.info('start {} epoch'.format(epoch))
    train_loss=[]
    time_num = 0
    update_time_now = 0
    for k in range(0, num_batch, 1):
      loss = 0
      optimizer.zero_grad()
      batch_idx = k

      if batch_idx >= num_batch:
        continue

      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
      edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
      timestamps_batch = train_data.timestamps[start_idx:end_idx]
      target_batch = train_data.labels[start_idx:end_idx]
      type_batch = train_data.types[start_idx:end_idx]

      size = len(sources_batch)


      cdgp.train()
      index = np.where(target_batch > 0)
      pred = cdgp.forward(sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, index, NUM_NEIGHBORS)

      if(sum(target_batch)>0):
        target_torch = torch.from_numpy(target_batch[index]).to(device)
        target = torch.log2(target_torch)
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        optimizer.zero_grad()
        loss = criterion(target, pred)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

      cdgp.memory.detach_memory()
      if USE_COMMUNITY:
          cdgp.community.detach_community2()
    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)
    ### Validation
    cdgp.set_neighbor_finder(full_ngh_finder)
    train_memory_backup = cdgp.memory.backup_memory()

    train_losses.append(np.mean(train_loss))
    logger.info('Epoch mean train loss: {}'.format(np.mean(train_loss)))
    val_results = eval_popularity_prediction(model=cdgp, criterion=criterion, data=val_data, n_neighbors=NUM_NEIGHBORS,
                                                                                    logger=logger, device=device)

    val_loss_list.append(val_results["loss"])
    val_msle_list.append(val_results["msle"])
    val_rmsle_list.append(val_results["rmsle"])
    val_male_list.append(val_results["male"])
    val_pcc_list.append(val_results["pcc"])
    val_mape_list.append(val_results["mape"])

    val_memory_backup = cdgp.memory.backup_memory()
    cdgp.memory.restore_memory(train_memory_backup)

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))


    # Early stopping
    if early_stopper.early_stop_check(val_results["rmsle"]):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      cdgp.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      cdgp.eval()
      break
    else:
      torch.save(cdgp.state_dict(), get_checkpoint_path(epoch))

  if USE_MEMORY:
    val_memory_backup = cdgp.memory.backup_memory()

  cdgp.embedding_module.neighbor_finder = full_ngh_finder
  test_results = eval_popularity_prediction(model=cdgp, criterion=criterion, data=test_data,
                                                                         n_neighbors=NUM_NEIGHBORS, logger=logger, device=device,
                                                                         type="test")

  cdgp.memory.restore_memory(val_memory_backup)
  plot_train(train_losses,val_loss_list,val_rmsle_list,val_msle_list,
             val_pcc_list,val_male_list,val_mape_list,args.prefix,i)

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    cdgp.memory.restore_memory(val_memory_backup)
  torch.save(cdgp.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')
