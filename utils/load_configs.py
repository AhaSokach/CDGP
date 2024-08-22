import argparse
import sys
import torch



def get_popularity_prediction_args():
    """
    get the args for the popularity prediction task
    :return:
    """
    # arguments for train
    parser = argparse.ArgumentParser('CDGP popularity prediction')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='aminer')
    parser.add_argument('--bs', type=int, default=100, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')

    # arguments for community
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user and community')
    parser.add_argument('--community_num', type=int, default=800, help='Number of communities')
    parser.add_argument('--community_layer', type=int, default=1, help='Number of community layers')
    parser.add_argument('--member_num', type=int, default=200, help='Number of community members')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args







