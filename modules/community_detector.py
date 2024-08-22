import math

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_scatter import scatter_add, scatter_max

from torch_geometric.nn import GCNConv
import torch_scatter
from modules.community_score_calculator import get_community_score_calculator
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_geometric.nn.pool.topk_pool import topk
import numpy as np


class CommunityDetector(torch.nn.Module):

    def __init__(self, in_channels, neighbor_finder, community_max_num, max_member_num, score_type, device, ratio=0.8,
                 dropout_att=0, negative_slope=0.2):
        super(CommunityDetector, self).__init__()

        self.in_channels = in_channels  # hidden
        self.max_member_num = max_member_num
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = 0
        self.neighbor_finder = neighbor_finder
        self.linear_max = Linear(in_channels, in_channels)
        self.linear_score = Linear(in_channels * 2, 1)

        self.community_score_calculator = get_community_score_calculator(score_type, in_channels, 1)
        self.community_max_num = community_max_num
        self.device = device
        self.visited = {}
        self.use_memory = True

    def forward(self, memory, node_features, community_embeddings, unique_node_ids, timestamps, community_score,
                community_index,
                member_score, node2community, community2node, member_num, community_layer=2,
                community_percent=[1, 0.05], community2link=None, link_score=None, link_num=None):

        neighbors, neighbors_noself, neighbor_bool, neighbor_bool_noself, unique_neighbors, neighbors_num, unique_source_nodes, source_nodes, index_source_nodes, \
            unique_neighbors_noself, source_nodes_noself, index_source_nodes_noself = self.neighbor_finder.get_community_neighbor(
            unique_node_ids,
            timestamps,
            max_member_num=self.max_member_num,
            )  # all_neighbor and itself

        neighbors_torch = torch.from_numpy(np.array(unique_neighbors)).long().to(self.device)  # neighbors_unique
        neighbors_torch = neighbors_torch.reshape(-1)
        unique_neighbors_noself = torch.from_numpy(np.array(unique_neighbors_noself)).long().to(self.device)
        neighbors_features = node_features[neighbors_torch, :] + memory.get_memory(neighbors_torch).detach() # TODO detach

        neighbor_bool = torch.tensor(neighbor_bool).to(self.device).bool()
        neighbor_bool_noself = torch.tensor(neighbor_bool_noself).to(self.device).bool()
        source_nodes = torch.tensor(source_nodes).long().to(self.device) # all source nodes corresponding to neighbors torch
        index_source_nodes = torch.tensor(index_source_nodes).long().to(self.device) # index of source nodes from zero
        unique_source_nodes = torch.tensor(unique_source_nodes).long().to(self.device) # unique source nodes
        index_source_nodes_noself = torch.tensor(index_source_nodes_noself).long().to(self.device)
        max_feature, _ = scatter_max(neighbors_features, index_source_nodes, dim=0)  # (index.size(), feature_dim) max pooling
        max_feature_linear = self.linear_max(max_feature)  #  (valid_nodes.size(), feature_dim)

        # max_feature[valid_nodes]=max_feature
        max_feature_duplication = max_feature_linear[index_source_nodes]
        center_feature_cat = torch.cat((max_feature_duplication, neighbors_features),
                                       dim=-1)  # (neighbors_size, feature_dim*2)
        score = self.linear_score(center_feature_cat)
        score = F.leaky_relu(score, self.negative_slope) # TODO
        score = softmax(src=score, index=source_nodes)
        score = F.dropout(score, p=self.dropout_att, training=self.training) # TODO

        node_num = len(neighbors_num)

        score_tensor = torch.zeros((node_num, self.max_member_num)).to(self.device)
        score_reshape = torch.reshape(score_tensor, (-1,))
        neighbor_bool_noself = neighbor_bool_noself[0:node_num].view(-1)
        neighbor_bool = neighbor_bool[0:node_num].view(-1)
        score_reshape[neighbor_bool] = score.view(-1)
        member_score[unique_source_nodes] = score_tensor

        # neighbors_tensor = torch.zeros((node_num, self.max_member_num)).to(self.device)
        # neighbors_tensor = torch.reshape(neighbors_tensor, (-1,))
        neighbors_noself = torch.tensor(neighbors_noself).to(self.device)
        neighbors = torch.tensor(neighbors).to(self.device)
        community2node[unique_source_nodes] = neighbors[0:node_num].long()
        member_num[unique_source_nodes] = (torch.tensor(neighbors_num).to(self.device) - 1).long()



        # score_tensor = score_tensor.scatter(1, index_source_nodes.view(1,-1), score.view(-1))

        # community2node[unique_source_nodes] = neighbors
        # community2link[unique_source_nodes] = neighbors
        # member_num[unique_source_nodes] = neighbors_num-1
        # link_num[unique_source_nodes] = neighbors_num-1
        # update new community structure for layer1
        # for i in range(len(neighbors_num)):  # update member score
        #     j = neighbors_num[i]
        #     n = unique_source_nodes[i]
        #     neigh_index = source_nodes == n
        #     neigh = neighbors_torch[neigh_index]
        #     neigh_score = score[neigh_index].view(-1)
        #     member_score[n][0:j-1] = neigh_score[0:j-1]
        #     community2node[n][0:j-1] = neigh[0:j-1]
        #     member_num[n]=neighbors_num[i]-1   # delete community center
        #     if (community_layer > 1):
        #         community2link[n][0:j-1] = neigh[0:j-1]
        #         link_score[n][0:j-1] = neigh_score[0:j-1]
        #         link_num[n] = neighbors_num[i] - 1

        if (community_layer > 1):
            index = []
            index1 = []
            all_valid_nodes_members = []
            all_valid_nodes_members_scores = []
            for i in range(len(unique_source_nodes)):
                n = unique_source_nodes[i]
                self.visited.clear()
                self.find_community_member(n, community2link, link_score, link_num, community_percent,
                                           n, community_layer, layer_now=0, score_now=1,
                                           max_member_num=self.max_member_num)
                node_member_num = len(list(self.visited.keys()))  # first layer
                community2node[n][0:node_member_num] = torch.from_numpy(np.array(list(self.visited.keys()))) \
                    .long().to(self.device)  # TODO max member
                member_score[n][0:node_member_num] = torch.from_numpy(np.array(list(self.visited.values()))) \
                    .to(self.device)
                index.extend([n] * node_member_num)
                index1.extend([i] * node_member_num)
                all_valid_nodes_members.extend(list(self.visited.keys()))
                all_valid_nodes_members_scores.extend(list(self.visited.values()))
                # TODO node2community?

        if (community_layer == 1):
            score_feature = neighbors_features * score.view(-1, 1)  # 119，172 * 119，1
            cluster_feature = scatter_add(score_feature, index_source_nodes, dim=0)  # (update node size , feature_dim)
            # cluster_feature = cluster_feature[index1]
            community_embeddings[unique_source_nodes] = cluster_feature  # all nodes's community feature
            # neighbors_community_embeddings = community_embeddings[neighbors_unique] # TODO these feature didnot update  maybe this place can update every cluster's embedding
            # gpu_tracker.track()

            fitness = torch.sigmoid(
                self.community_score_calculator.calculate(community_embeddings, memory, unique_source_nodes, index=source_nodes, index1=index_source_nodes,
                                                neighbors_unique=unique_neighbors, index_noself=source_nodes_noself, index1_noself= index_source_nodes_noself,
                                                          neighbors_unique_noself=unique_neighbors_noself)).view(-1)  # score for every cluster
        else:
            # member_nodes = community2node
            index = torch.tensor(index).long().to(self.device)
            index1 = torch.tensor(index1).long().to(self.device)
            member_feature = memory.get_memory(all_valid_nodes_members).detach()  # TODO why
            member_feature = member_feature + node_features[all_valid_nodes_members]
            all_valid_nodes_members_scores_tensor = torch.from_numpy(
                np.array(all_valid_nodes_members_scores)).float().to(self.device)
            score_feature = member_feature * all_valid_nodes_members_scores_tensor.view(-1, 1)
            cluster_feature = scatter_add(score_feature, index1, dim=0)
            community_embeddings[unique_source_nodes] = cluster_feature
            fitness = torch.sigmoid(
                self.community_score_calculator.calculate(community_embeddings, unique_source_nodes, index=index, index1=index1,
                                                neighbors_unique=all_valid_nodes_members_scores_tensor)).view(-1)

        # score_cluster_featurev= torch.matmul(fitness.view(1,-1), cluster_feature)

        return_fit = fitness
        if (community_score.shape[0] != 0):
            fitness = torch.cat((fitness, community_score))
            community_index = torch.cat((unique_source_nodes, community_index))
            unique_community_index, unique_indices = torch.unique(community_index, return_inverse=True)
            unique_fitness = torch.zeros(unique_community_index.shape[0]).to(self.device)
            unique_fitness[unique_indices] = fitness

            if (unique_community_index.shape[0] < self.community_max_num):
                community_score = unique_fitness
                community_index = unique_community_index
            else:
                perm, indices = unique_fitness.topk(k=self.community_max_num,
                                                    largest=True)  # (x=fitness, ratio=self.ratio) # bao liu jie dian de bi li
                community_score = unique_fitness[indices]  # perm is index
                community_index = unique_community_index[indices]
        #     fitness = torch.cat((fitness, community_score))
        #     community_index = torch.cat((unique_source_nodes, community_index))
        #     # unique_community_index, unique_indices = torch.unique(community_index, return_inverse=True)
        #     # unique_fitness = torch.zeros(unique_community_index.shape[0]).to(self.device)
        #     # unique_fitness[unique_indices] = fitness
        #
        #     if (community_index.shape[0] < self.community_max_num):
        #         community_score = fitness
        #         new_community_index = community_index
        #     else:
        #         perm, indices = fitness.topk(k=self.community_max_num,
        #                                             largest=True)  # (x=fitness, ratio=self.ratio) # bao liu jie dian de bi li
        #         community_score = fitness[indices]  # perm is index
        #         new_community_index = torch.lt(indices, unique_source_nodes.shape[0])
        #         new_community_index = indices[new_community_index]
        #         new_community_index = community_index[new_community_index]
        #         community_index = community_index[indices]
        else:
            community_index = unique_source_nodes
            community_score = fitness
        #     new_community_index = community_index
        # calculate jiao ji between perm and index

        # update node2community TODO change newly involved node
        if community_layer == 1:
            if neighbors_torch.shape[0] != 0:
                true_index = (source_nodes == community_index.view(-1, 1)).any(dim=0)
                change_index_neighbor = neighbors_torch[true_index]
                node2community[change_index_neighbor] = source_nodes[true_index]
        else:
            if (len(all_valid_nodes_members) != 0):
                node2community[all_valid_nodes_members] = index
        return community_embeddings, community_score, community_index, member_score, node2community, community2node, member_num

    def find_community_member(self, root, community2link, link_score, link_num, community_percent,
                              new_center, layer, layer_now, score_now, max_member_num):
        if (layer_now + 1 == layer):
            return
        percent = community_percent[layer_now]
        center_member_num = link_num[new_center]
        center_member_all = community2link[new_center][:center_member_num]
        center_member_score = link_score[new_center][:center_member_num]
        layer_member_num = int(center_member_num * percent)  # choose percent * num nodes as members
        layer_top_score, layer_indices = center_member_score.topk(
            k=layer_member_num, largest=True
        )
        layer_member = center_member_all[layer_indices]
        layer_member_score = layer_top_score
        for j in range(len(layer_member)):
            if (len(self.visited) == max_member_num):
                return
            m = int(layer_member[j].item())
            s = float(layer_member_score[j].item())
            score_old = self.visited.get(m)
            if (m != root and (score_old == None or score_old < score_now * s)):
                if (score_old != None):
                    self.visited.pop(m)
                self.visited[m] = score_now * s
                if (len(self.visited) + len(layer_member) - j - 1 <= max_member_num):
                    self.find_community_member(root, community2link, link_score, link_num, community_percent,
                                               m, layer, layer_now + 1, score_now * s, max_member_num)
