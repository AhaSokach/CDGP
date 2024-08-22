import numpy as np
import torch
import matplotlib.pyplot as plt
import random

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)

class MLP_edge(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, dim//2)
        self.fc_2 = torch.nn.Linear(dim//2, dim//4)
        self.fc_3 = torch.nn.Linear(dim//4, dim//8)
        self.fc_4 = torch.nn.Linear(dim // 8, 1)
        self.act = torch.nn.ReLU()
        self.dim=dim
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        # print(self.dim)
        # print(x.shape)
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        x = self.act(self.fc_3(x))
        x = self.dropout(x)
        return self.fc_4(x).sigmoid().squeeze(dim=1)

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):  # adj_list for every node
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

        self.uniform = uniform
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        # print("src_idx")
        # print(src_idx)
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[
                                                                                             src_idx][:i]

    def get_community_neighbor(self, source_nodes, timestamps, max_member_num=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = max_member_num if max_member_num > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), max_member_num)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        neighbors_noself = np.zeros((len(source_nodes), max_member_num)).astype(
            np.int32)
        neighbor_bool = np.zeros((len(source_nodes), max_member_num)).astype(
            np.int32)
        neighbor_bool_noself = np.zeros((len(source_nodes), max_member_num)).astype(
            np.int32)
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        unique_neighbors = []
        unique_neighbors_noself = []
        neighbors_num = []
        unique_source_nodes = []
        dup_source_nodes = []
        dup_source_nodes_noself = []
        index_source_nodes = []
        index_source_nodes_noself = []
        j = 0
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            # print(timestamp)
            timestamp = timestamp.cpu().numpy()
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                                                     timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and max_member_num > 0:
                source_neighbors = source_neighbors.tolist()[-max_member_num+1:]
                # source_neighbors = np.flip(source_neighbors)
                source_neighbors = np.unique(source_neighbors).tolist()
                unique_neighbors_noself.extend(source_neighbors[-max_member_num + 1:])
                source_neighbors.append(source_node)
                unique_neighbors.extend(source_neighbors[-max_member_num:])

                #unique_members.extend(source_neighbors[-max_member_num-1:])
                num = min(len(source_neighbors), max_member_num)
                neighbors_num.append(num)
                unique_source_nodes.append(source_node)
                dup_source_nodes.extend(num * [source_node]) # corresponding to nunique eighbors
                dup_source_nodes_noself.extend((num - 1) * [source_node])

                index_source_nodes.extend(num * [j]) # corresponding to dup_source_nodes
                index_source_nodes_noself.extend((num-1) * [j])
                neighbors[j, 0:num] = source_neighbors[-num:]
                neighbors_noself[j, 0:num-1] = source_neighbors[-num+1:]
                neighbor_bool[j,0:num] = 1
                neighbor_bool_noself[j, 0:num-1] = 1

                #source_edge_times = source_edge_times[-max_member_num:]
                #source_neighbors = source_neighbors[-max_member_num:]
                #source_edge_idxs = source_edge_idxs[-n_neighbors:]

                #edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                #edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
                j += 1

        return neighbors,neighbors_noself,neighbor_bool, neighbor_bool_noself, unique_neighbors, neighbors_num, unique_source_nodes,dup_source_nodes,index_source_nodes, \
                unique_neighbors_noself, dup_source_nodes_noself, index_source_nodes_noself

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                                                     timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times

def plot_train(train_losses, val_loss_list, val_rmsle_list, val_msle_list,
               val_pcc_list, val_male_list, val_mape_list, name, run):
    fig = plt.figure()
    plt.subplot(321)
    x = np.arange(len(train_losses))
    plt.plot(x, train_losses, label='train_loss')
    plt.title('loss')
    plt.subplot(322)
    plt.plot(x, val_rmsle_list, label=f'val_rmsle')
    # axis.set_title('rmse')
    plt.title('rmsle')
    plt.subplot(323)
    plt.plot(x, val_msle_list, label=f'val_msle')
    plt.title('msle')
    plt.subplot(324)
    plt.plot(x, val_pcc_list, label=f'val_pcc')
    plt.title('pcc')
    plt.subplot(325)
    plt.plot(x, val_male_list, label=f'val_male')
    plt.title('male')
    '''plt.subplot(236) 
    plt.plot(x, val_mape_list, label=f'val_mape')
    plt.title('mape')'''
    plt.suptitle(f'runs_{run}')
    # plt.tight_layout()
    plt.savefig(f'fig/{name}_{run}.png')
