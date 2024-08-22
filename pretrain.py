import math
import logging
import random
import time
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import inspect

from evaluation.evaluation import eval_popularity_prediction, eval_edge_prediction
from model.cdgp import CDGP
from utils.utils import EarlyStopMonitor, set_random_seed, RandEdgeSampler, get_neighbor_finder, plot_train
from utils.data_processing import get_data_popularity, compute_time_statistics
from utils.load_configs import get_popularity_prediction_args

args = get_popularity_prediction_args()
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# nohup python train.py -d aminer --lr 0.00005 --gpu 0 > aminer.log 2>&1 &

### Argument and global variables
# parser = argparse.ArgumentParser('CDGP popularity prediction')
# parser.add_argument('-d', '--data', type=str, help='Dataset name',
#                     default='aminer')
# parser.add_argument('--bs', type=int, default=100, help='Batch_size')
# parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
# parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
# parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
# parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
# parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
# parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
# parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
# parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
# parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
# parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
# parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
#
# parser.add_argument('--community_updater', type=str, default="topk", choices=[
#     "topk", "kmeans"], help='Type of memory updater')
# parser.add_argument('--embedding_module', type=str, default="identity", choices=[
#     "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
# parser.add_argument('--message_function', type=str, default="identity", choices=[
#     "mlp", "identity"], help='Type of message function')
# parser.add_argument('--memory_updater', type=str, default="gru", choices=[
#     "gru", "rnn"], help='Type of memory updater')
# parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
#                                                                    'aggregator')
# parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
# parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
#                                                                 'each user')
# parser.add_argument('--community_num', type=int, default=800, help='community_num')
# parser.add_argument('--community_layer', type=int, default=1, help='community_layer')
# parser.add_argument('--community_score', type=str, default="community", help='Type of community'
#                                                                              'score')
# parser.add_argument('--member_num', type=int, default=200, help='member_num')
#
# try:
#     args = parser.parse_args()
# except:
#     parser.print_help()
#     sys.exit(0)

BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
LEARNING_RATE = args.lr
MEMORY_DIM = args.memory_dim
COMMUNITY_NUM = args.community_num
COMMUNITY_LAYER = args.community_layer
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

# Get data for training, validation and testing
full_data, train_data, val_data, test_data, node_features, edge_features, n_nodes, n_edges = get_data_popularity(DATA)

# Initialize neighbor finder
train_ngh_finder = get_neighbor_finder(train_data)
full_ngh_finder = get_neighbor_finder(full_data)


train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
# device_string='cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

frame = inspect.currentframe()

for i in range(args.n_runs):
    set_random_seed(i)
    Path("results/").mkdir(parents=True, exist_ok=True)
    dis_path = f'./test/weight-{args.prefix}-{i}-{DATA}.txt'
    dis_file = open(dis_path, "w")
    # Initialize Model
    cdgp = CDGP(neighbor_finder=full_ngh_finder, node_features=node_features,
                edge_features=edge_features,
                device=device,
                n_nodes=n_nodes,
                n_edges=n_edges,
                dropout=DROP_OUT,
                # community_updater_type=COMMUNITY_UPDATER_TYPE,
                community_num=COMMUNITY_NUM,
                # community_score_type=args.community_score,
                community_layer=COMMUNITY_LAYER,
                member_num=MEMBER_NUM,
                memory_dimension=MEMORY_DIM,
                community_dimension=MEMORY_DIM,
                # embedding_module_type=args.embedding_module,
                # message_function=args.message_function,
                # aggregator_type=args.aggregator,
                # memory_updater_type=args.memory_updater,
                # n_neighbors=NUM_NEIGHBORS,
                # mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                # mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                file = dis_file)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(cdgp.parameters(), lr=LEARNING_RATE)
    cdgp = cdgp.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    epoch_times = []
    total_epoch_times = []
    train_losses = []
    train_loss_list = []
    val_loss_list = []
    val_rmsle_list = []
    val_msle_list = []
    val_pcc_list = []
    val_male_list = []
    val_mape_list = []

    batch_community_index_list = []

    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize representation of the model at the start of each epoch
        cdgp.memory.__init_memory__()
        cdgp.community.__init_community__()
        # cdgp.community_memory_updater.__init_weight__()
        logger.info('start {} epoch'.format(epoch))
        train_loss = []
        back_num = 0


        for batch in range(0, num_batch, 1):
            # print(f'batch:{batch}')
            # loss = 0.0
            optimizer.zero_grad()

            # optimizer.zero_grad()
            start_idx = batch * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                train_data.destinations[start_idx:end_idx]
            edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = train_data.timestamps[start_idx:end_idx]
            target_batch = train_data.labels[start_idx:end_idx]
            type_batch = train_data.types[start_idx:end_idx]

            size = len(sources_batch)
            _, negatives_batch = train_rand_sampler.sample(size)
            # print(target_batch)

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            cdgp = cdgp.train()
            # index = np.where(target_batch > 0)
            # if (sum(target_batch) > 0) and ((batch % 5) == 0):
            #     optimizer.zero_grad()
            score_pos, score_neg = cdgp.forward_edge(sources_batch, destinations_batch,negatives_batch, timestamps_batch,
                                                     edge_idxs_batch
                                                     )

            # print(score_pos.squeeze().shape)
            # print(pos_label.shape)

            loss = criterion(score_pos.squeeze(), pos_label) + criterion(score_neg.squeeze(), neg_label)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())
            cdgp.memory.detach_memory()
            cdgp.community.detach_community()


        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        # print("train end")

        train_losses.append(np.mean(train_loss))
        logger.info('Epoch mean train loss: {}'.format(np.mean(train_loss)))

        ### Validation
        # cdgp.set_neighbor_finder(full_ngh_finder)
        ap, acc = eval_edge_prediction(model=cdgp, negative_edge_sampler=val_rand_sampler,
                                                                                data=val_data,
                                                                                # n_neighbors=NUM_NEIGHBORS,

                                                                                 batch_size=BATCH_SIZE )
        logger.info(f'ap:{ap} acc:{acc}')
        # val_loss_list.append(val_results["loss"])
        # val_msle_list.append(val_results["msle"])
        # val_rmsle_list.append(val_results["rmsle"])
        # val_male_list.append(val_results["male"])
        # val_pcc_list.append(val_results["pcc"])
        # val_mape_list.append(val_results["mape"])

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))

        # Early stopping
        # print("save1")
        if early_stopper.early_stop_check(ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            cdgp.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            cdgp.eval()
            break
        else:
            torch.save(cdgp.state_dict(), get_checkpoint_path(epoch))
        # print("save2")


    ### Test
    # cdgp.set_neighbor_finder(full_ngh_finder)
    ap, acc = eval_edge_prediction(model=cdgp, negative_edge_sampler=test_rand_sampler,
                                                                                data=test_data,
                                                                                # n_neighbors=NUM_NEIGHBORS,

                                                                                 batch_size=BATCH_SIZE )

    # plot_train(train_losses, val_loss_list, val_rmsle_list, val_msle_list,
    #            val_pcc_list, val_male_list, val_mape_list, args.prefix, i)
    logger.info(f'ap:{ap} acc:{acc}')
    logger.info('Saving cdgp model')
    torch.save(cdgp.state_dict(), MODEL_SAVE_PATH)
    logger.info('cdgp model saved')
