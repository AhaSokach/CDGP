import math

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_scatter import scatter_add, scatter_max

from torch_geometric.nn import GCNConv
import torch_scatter
from modules.le_conv import get_community_score_calculator
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_geometric.nn.pool.topk_pool import topk
import numpy as np

from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm


# torch.set_num_threads(1)

class Pooling(torch.nn.Module):

    def __init__(self, in_channels, neighbor_finder, community_max_num, max_member_num, score_type, device, ratio= 0.8, dropout_att=0, negative_slope=0.2):
        super(Pooling, self).__init__()

        self.in_channels = in_channels #hidden
        self.max_member_num = max_member_num
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = 0
        self.neighbor_finder = neighbor_finder
        self.linear_max = Linear(in_channels, in_channels)
        self.linear_score = Linear(in_channels*2, 1)

        self.community_score_calculator = get_community_score_calculator(score_type,in_channels, 1)
        #self.leconv = LEConv(in_channels, 1)
        #self.gat_att = Linear(2*in_channels, 1)
        #self.gnn_score = LEConv(self.in_channels, 1) # gnn_score: uses LEConv to find cluster fitness scores
        # self.gnn_intra_cluster = GCNConv(self.in_channels, self.in_channels) # gnn_intra_cluster: uses GCN to account for intra cluster properties, e.g., edge-weights
        # self.reset_parameters()
        self.community_max_num = community_max_num
        self.device = device
        self.use_memory = True
        


    '''def reset_parameters(self):

        self.lin_q.reset_parameters()
        self.gat_att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.gnn_intra_cluster.reset_parameters()'''
        
             
    def forward(self, memory, node_features, community_embeddings ,unique_node_ids,  timestamps, community_score, community_index,
                member_score,avg_member_score, node2community, community2node, member_num,community_layer=1,community_percent = None,
                node2community_layer=None, community2node_layer=None, member_num_layer=None, #new_community_information=None,
                n_neighbors=20, time_diffs=None, use_time_proj=True):

        #gpu_tracker.track()


        #nodes_torch = torch.from_numpy(nodes).long().to(self.device)  # update nodes
        #timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        #nodes_time_embedding = self.time_encoder(torch.zeros_like(
        #    timestamps_torch))

        # find max pool in every update node and their neighbors
        '''dynamic_new_center=[]
        dynamic_new_center_embeddings=[]
        dynamic_new_times = []'''
        neighbors, edge_idxs, edge_times,neighbors_unique,neighbors_num, valid_nodes = self.neighbor_finder.get_community_neighbor(
            unique_node_ids,
            timestamps,
            max_member_num=self.max_member_num,
            n_neighbors=n_neighbors) # all_neighbor and itself

        neighbors_unique = np.array(neighbors_unique)

        neighbors_torch = torch.from_numpy(neighbors_unique).long().to(self.device) # neighbors_unique
        neighbors_torch = neighbors_torch.reshape(-1)


        neighbors_features = node_features[neighbors_torch, :] # unique neibor
        #gpu_tracker.track()

        if self.use_memory:
            memory_neighbors = memory.get_memory(neighbors_torch).detach() # TODO tensor [neighbor_num, 1]
            neighbors_features = memory_neighbors + neighbors_features # TODO

        index = []
        j = 0
        index1 = []
        for node in valid_nodes:
            for i in range(neighbors_num[j]):
                index.append(node)
                index1.append(j)
            j+=1
        #index = [node for node in valid_nodes for i in range(neighbors_num[i])] #every node's neighbor num not same
        # index is edge index 0
        #print(index)

        index = torch.tensor(index).long().to(self.device)
        index1 = torch.tensor(index1).long().to(self.device)

        valid_nodes = torch.tensor(valid_nodes).long().to(self.device)
        max_feature, _ = scatter_max(neighbors_features,index1,dim=0) # (index.size(), feature_dim)
        max_feature_linear = self.linear_max(max_feature)# TODO (valid_nodes.size(), feature_dim)  every node's cluster's feature
        # chongfu 10 bian
        #max_feature[valid_nodes]=max_feature
        max_feature_duplication = max_feature_linear[index1]
        center_feature_cat = torch.cat((max_feature_duplication,neighbors_features),dim=-1) # TODO (neighbors_size, feature_dim*2)
        score = self.linear_score(center_feature_cat) # calculate every member's lushudu of cluster
        # TODO neighbor chongfule every member of cluster'score of
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(src=score, index=index) # TODO
        #score = score[index]
        score = F.dropout(score, p=self.dropout_att, training=self.training)

        score_feature =  neighbors_features * score.view(-1, 1)  # 119，172 * 119，1
        cluster_feature = scatter_add(score_feature, index1, dim = 0) #(update node size , feature_dim)
        #cluster_feature = cluster_feature[index1]
        #community_embeddings[valid_nodes] = cluster_feature  # all nodes's community feature
        #neighbors_community_embeddings = community_embeddings[neighbors_unique] # TODO these feature didnot update  maybe this place can update every cluster's embedding
        #gpu_tracker.track()

        fitness = torch.sigmoid(self.community_score_calculator(community_embeddings,valid_nodes, index = index, index1 = index1,neighbors_unique = neighbors_unique)).view(-1) # score for every cluster
        #gpu_tracker.track()
        #score_cluster_featurev= torch.matmul(fitness.view(1,-1), cluster_feature)
        community_embeddings[valid_nodes] = cluster_feature
        return_fit = fitness
        #community_embeddings[valid_nodes] = torch.matmul(fitness.view(1,-1) , cluster_feature ) # all nodes's community feature

        for i in range(len(neighbors_num)): # update member score
            j = neighbors_num[i]
            n = valid_nodes[i]
            neigh_index = index == n
            neigh = neighbors_torch[neigh_index]
            community2node[n][0:j] = neigh
            neigh_score = score[neigh_index].view(-1)
            member_score[n][0:j] = neigh_score
            member_num[n] = j

        if (community_layer > 1):
            new_center = valid_nodes
            center_member_num = neighbors_num # first layer
            center_member_score = score
            layer_score = 0
            percent = community_percent[0]
            for j in range(len(valid_nodes)):  # loop for all valid nodes - one hop
                layer_member_num = int((center_member_num[j]) * percent)
                layer_top_score, layer_indices = member_score[new_center[j]][:center_member_num[j]].topk(k=layer_member_num)
                layer_member = community2node[new_center[j]][layer_indices] #one hop neighbor
                second_layer_member_all = community2node[layer_member]
                second_layer_score_all = member_score[layer_member]
                second_layer_num_all = member_num[layer_member]
                layer_score = 0
                add_community_member = []
                percent = community_percent[1]
                for z in range(len(layer_member)): # loop for one hop neighbor - find 2 hop
                    if(layer_member[z] in add_community_member):
                        continue
                    second_layer_member_num = int((second_layer_num_all[z]) * percent)
                    second_layer_top_score, second_layer_indices = second_layer_score_all[z][:second_layer_num_all[z]].topk(k=second_layer_member_num)
                    second_layer_member = second_layer_member_all[z][second_layer_indices]
                    second_layer_member_score = second_layer_score_all[z][second_layer_indices]
                    layer_score += torch.matmul(second_layer_member_score, layer_top_score[z].repeat(second_layer_member_num,1))
                    add_community_member.extend(second_layer_member.tolist())
                    if(community_layer>2):
                        third_layer_member_all = community2node[second_layer_member]
                        third_layer_score_all = member_score[second_layer_member]
                        third_layer_num_all = member_num[second_layer_member]
                        percent = community_percent[2]
                        for m in range(second_layer_member_num):
                            if(second_layer_member[m] in add_community_member):
                                continue
                            third_layer_member_num = int((third_layer_num_all[m]) * percent)
                            third_layer_top_score, third_layer_indices = third_layer_score_all[m][:third_layer_num_all[m]].topk(
                                k=third_layer_member_num)
                            third_layer_member = third_layer_member_all[m][third_layer_indices]
                            third_layer_member_score = third_layer_score_all[m][third_layer_indices]
                            layer_score += torch.matmul(third_layer_member_score, second_layer_member_score[m].repeat(third_layer_member_num,1))
                            add_community_member.extend(third_layer_member.tolist())
                fitness[j] = fitness[j] + layer_score
                #print(f'layer_score:{layer_score}')
                add_community_member = np.unique(np.array(add_community_member)).tolist()
                add_len = len(add_community_member)
                community2node_layer[new_center[j]][0:add_len] = torch.tensor(add_community_member).to(self.device)
                member_num_layer[new_center[j]] = add_len
                node2community_layer[add_community_member] = new_center[j]


        #fitness = [fitness, community_score]
        #unique_node_ids = torch.tensor(unique_node_ids).to(self.device)
        if(community_score.shape[0]!=0):
            fitness = torch.cat((fitness, community_score))
            community_index = torch.cat((valid_nodes, community_index))
            #community_index, ri = torch.unique(input=community_index,return_inverse=True,sorted=False) # if there is same id
            #fitness = fitness[ri] # delete chongfu
            #[unique_node_ids, community_index]
            if(community_index.shape[0] < self.community_max_num):
                community_score = fitness
                new_community_index = community_index
            else:
                perm, indices = fitness.topk(k=self.community_max_num,
                                             largest=True)  # (x=fitness, ratio=self.ratio) # bao liu jie dian de bi li
                community_score = fitness[indices]  # perm is index
                new_community_index = torch.lt(indices, valid_nodes.shape[0])
                new_community_index = indices[new_community_index]
                new_community_index = community_index[new_community_index]
                community_index = community_index[indices]
        else:
            community_index = valid_nodes
            community_score = fitness
            new_community_index = community_index
        # calculate jiao ji between perm and index
        '''neighbors = torch.from_numpy(neighbors)
        neighbors = neighbors.to(self.device)
        neighbors = torch.index_select(neighbors,
                                       index=torch.linspace(0,valid_nodes.shape[0]-1,valid_nodes.shape[0], dtype=torch.int).to(self.device),
                                       dim=0)'''



        #new_comunity = community_index[(community_index == valid_nodes.view(-1, 1)).any(dim=0)]
        #neighbors_community = torch.zeros((119, ), dtype=torch.int64).to(self.device)
        #neighbors_community = torch.scatter(input=neighbors_community, dim=0, index=index, src=neighbors_torch)
        #neighbors_community[valid_nodes] = neighbors

        #community2node[valid_nodes] = neighbors
        #print("community_score:")
        #print(community_score)
        '''
        neighbors = torch.zeros(max_feature.shape[0], self.max_member_num).long().to(self.device)
        neighbors = torch.scatter(neighbors, 0, index.view(1, -1), neighbors_torch.view(1, -1))
        neighbors = neighbors[new_comunity]  # not from index 0
        community2node[new_comunity] = neighbors

        neighbors_score = torch.zeros(max_feature.shape[0], self.max_member_num).float().to(self.device)
        neighbors_score = torch.scatter(neighbors_score, 0, index.view(1,-1), score.view(1,-1))
        neighbors_score = neighbors_score[new_comunity] # not from index 0
        print(neighbors_score[0])
        #print(neighbors_score[1]) 
        member_score[new_comunity] = neighbors_score'''
        new_index = (index == new_community_index.view(-1, 1)).any(dim=0)
        neighbors_new = neighbors_torch[new_index]
        new_index2 = index[new_index]
        node2community[neighbors_new] = new_index2 # TODO
        member_num[valid_nodes] = torch.tensor(neighbors_num).long().to(self.device) # same length with valid nodes
        #print(member_num.dtype)
        #print(valid_nodes.dtype)
        if(community_layer>1):
            other_index = ~new_index
            neighbors_other = neighbors_torch[other_index]
            node2community[neighbors_other] = node2community_layer[neighbors_other]





        del neighbors_torch

        # TODO edge weight





           #src_list = [] TODO have double links



        # NxF
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Add Self Loops TODO ?
        #fill_value = 1
        #num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        #edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_weight=edge_weight,
        #    fill_value=fill_value, num_nodes=num_nodes.sum()) # add self loop for all nodes??

        #N = x.size(0) # total num of nodes in batch

        # ExF why gcn again TODO
        #x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index, edge_weight=edge_weight) # TODO why?
        #x_pool_j = x_pool[edge_index[1]] # all dst
        #x_j = x[edge_index[1]]'''
        
        #---Master query formation---
        #gpu_tracker.track()
        if(community_layer > 1):
            return community_embeddings, community_score, community_index, member_score,  node2community, community2node, member_num, \
               community2node_layer, node2community_layer, member_num_layer
        else:
            return community_embeddings, community_score, community_index, member_score,avg_member_score, node2community, community2node, member_num,return_fit

    def forward3(self, memory, node_features, community_embeddings, unique_node_ids, timestamps, community_score,
                community_index,
                member_score, avg_member_score,
                 node2community, community2node, member_num, community_layer=1, community_percent=None,
                node2community_layer=None, community2node_layer=None, member_num_layer=None,
                # new_community_information=None,
                n_neighbors=20, time_diffs=None, use_time_proj=True):
    # for true community index
        # gpu_tracker.track()

        # nodes_torch = torch.from_numpy(nodes).long().to(self.device)  # update nodes
        # timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        # nodes_time_embedding = self.time_encoder(torch.zeros_like(
        #    timestamps_torch))

        # find max pool in every update node and their neighbors
        '''dynamic_new_center=[]
        dynamic_new_center_embeddings=[]
        dynamic_new_times = []'''
        neighbors, edge_idxs, edge_times, neighbors_unique, neighbors_num, valid_nodes = self.neighbor_finder.get_community_neighbor(
            unique_node_ids,
            timestamps,
            max_member_num=self.max_member_num,
            n_neighbors=n_neighbors)  # all_neighbor and itself

        neighbors_unique = np.array(neighbors_unique)

        neighbors_torch = torch.from_numpy(neighbors_unique).long().to(self.device)  # neighbors_unique
        neighbors_torch = neighbors_torch.reshape(-1)

        neighbors_features = node_features[neighbors_torch, :]  # unique neibor
        # gpu_tracker.track()

        if self.use_memory:
            memory_neighbors = memory.get_memory(neighbors_torch).detach()  # TODO tensor [neighbor_num, 1]
            neighbors_features = memory_neighbors + neighbors_features  # TODO

        index = []
        j = 0
        index1 = []
        for node in valid_nodes:
            for i in range(neighbors_num[j]):
                index.append(node)
                index1.append(j)
            j += 1
        # index = [node for node in valid_nodes for i in range(neighbors_num[i])] #every node's neighbor num not same
        # index is edge index 0
        # print(index)

        index = torch.tensor(index).long().to(self.device)
        index1 = torch.tensor(index1).long().to(self.device)

        valid_nodes = torch.tensor(valid_nodes).long().to(self.device)
        max_feature, _ = scatter_max(neighbors_features, index1, dim=0)  # (index.size(), feature_dim)
        max_feature_linear = self.linear_max(
            max_feature)  # TODO (valid_nodes.size(), feature_dim)  every node's cluster's feature
        # chongfu 10 bian
        # max_feature[valid_nodes]=max_feature
        max_feature_duplication = max_feature_linear[index1]
        center_feature_cat = torch.cat((max_feature_duplication, neighbors_features),
                                       dim=-1)  # TODO (neighbors_size, feature_dim*2)
        score = self.linear_score(center_feature_cat)  # calculate every member's lushudu of cluster
        # TODO neighbor chongfule every member of cluster'score of
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(src=score, index=index)  # TODO
        # score = score[index]
        score = F.dropout(score, p=self.dropout_att, training=self.training)

        score_feature = neighbors_features * score.view(-1, 1)  # 119，172 * 119，1
        cluster_feature = scatter_add(score_feature, index1, dim=0)  # (update node size , feature_dim)
        # cluster_feature = cluster_feature[index1]
        # community_embeddings[valid_nodes] = cluster_feature  # all nodes's community feature
        # neighbors_community_embeddings = community_embeddings[neighbors_unique] # TODO these feature didnot update  maybe this place can update every cluster's embedding
        # gpu_tracker.track()

        fitness = torch.sigmoid(
            self.community_score_calculator(community_embeddings, valid_nodes, index=index, index1=index1,
                                            neighbors_unique=neighbors_unique)).view(-1)  # score for every cluster
        # gpu_tracker.track()
        # score_cluster_featurev= torch.matmul(fitness.view(1,-1), cluster_feature)
        community_embeddings[valid_nodes] = cluster_feature
        return_fit = fitness
        # community_embeddings[valid_nodes] = torch.matmul(fitness.view(1,-1) , cluster_feature ) # all nodes's community feature

        for i in range(len(neighbors_num)):  # update member score
            j = neighbors_num[i]
            n = valid_nodes[i]
            #if n == 25507:
            #    print("find")
            neigh_index = index == n
            neigh = neighbors_torch[neigh_index]
            community2node[n][0:j] = neigh
            neigh_score = score[neigh_index].view(-1)
            member_score[n][0:j] = neigh_score
            avg_member_score[n][0:j] = neigh_score * j
            member_num[n] = j

        if (community_layer > 1):
            new_center = valid_nodes
            center_member_num = neighbors_num  # first layer
            center_member_score = score
            layer_score = 0
            percent = community_percent[0]
            for j in range(len(valid_nodes)):  # loop for all valid nodes - one hop
                layer_member_num = int((center_member_num[j]) * percent)
                layer_top_score, layer_indices = member_score[new_center[j]][:center_member_num[j]].topk(
                    k=layer_member_num)
                layer_member = community2node[new_center[j]][layer_indices]  # one hop neighbor
                second_layer_member_all = community2node[layer_member]
                second_layer_score_all = member_score[layer_member]
                second_layer_num_all = member_num[layer_member]
                layer_score = 0
                add_community_member = []
                percent = community_percent[1]
                for z in range(len(layer_member)):  # loop for one hop neighbor - find 2 hop
                    if (layer_member[z] in add_community_member):
                        continue
                    second_layer_member_num = int((second_layer_num_all[z]) * percent)
                    second_layer_top_score, second_layer_indices = second_layer_score_all[z][
                                                                   :second_layer_num_all[z]].topk(
                        k=second_layer_member_num)
                    second_layer_member = second_layer_member_all[z][second_layer_indices]
                    second_layer_member_score = second_layer_score_all[z][second_layer_indices]
                    layer_score += torch.matmul(second_layer_member_score,
                                                layer_top_score[z].repeat(second_layer_member_num, 1))
                    add_community_member.extend(second_layer_member.tolist())
                    if (community_layer > 2):
                        third_layer_member_all = community2node[second_layer_member]
                        third_layer_score_all = member_score[second_layer_member]
                        third_layer_num_all = member_num[second_layer_member]
                        percent = community_percent[2]
                        for m in range(second_layer_member_num):
                            if (second_layer_member[m] in add_community_member):
                                continue
                            third_layer_member_num = int((third_layer_num_all[m]) * percent)
                            third_layer_top_score, third_layer_indices = third_layer_score_all[m][
                                                                         :third_layer_num_all[m]].topk(
                                k=third_layer_member_num)
                            third_layer_member = third_layer_member_all[m][third_layer_indices]
                            third_layer_member_score = third_layer_score_all[m][third_layer_indices]
                            layer_score += torch.matmul(third_layer_member_score,
                                                        second_layer_member_score[m].repeat(third_layer_member_num, 1))
                            add_community_member.extend(third_layer_member.tolist())
                fitness[j] = fitness[j] + layer_score
                # print(f'layer_score:{layer_score}')
                add_community_member = np.unique(np.array(add_community_member)).tolist()
                add_len = len(add_community_member)
                community2node_layer[new_center[j]][0:add_len] = torch.tensor(add_community_member).to(self.device)
                member_num_layer[new_center[j]] = add_len
                node2community_layer[add_community_member] = new_center[j]

        # fitness = [fitness, community_score]
        # unique_node_ids = torch.tensor(unique_node_ids).to(self.device)
        if (community_score.shape[0] != 0):
            fitness = torch.cat((fitness, community_score))
            community_index = torch.cat((valid_nodes, community_index))
            unique_community_index,unique_indices = torch.unique(community_index,return_inverse=True)
            unique_fitness = torch.zeros(unique_community_index.shape[0]).to(self.device)
            unique_fitness[unique_indices] = fitness

            if (unique_community_index.shape[0] < self.community_max_num):
                community_score = unique_fitness
                community_index = unique_community_index
            else:
                perm, indices = unique_fitness.topk(k=self.community_max_num,
                                             largest=True)  # (x=fitness, ratio=self.ratio) # bao liu jie dian de bi li
                community_score = unique_fitness[indices]  # perm is index
                '''new_community_index = torch.lt(indices, valid_nodes.shape[0])
                new_community_index = indices[new_community_index]
                new_community_index = community_index[new_community_index]'''
                community_index = unique_community_index[indices]
        else:
            community_index = valid_nodes
            community_score = fitness
            #new_community_index = community_index
        # calculate jiao ji between perm and index
        if neighbors_torch.shape[0] != 0:
            node_community_before = node2community[neighbors_torch]

            true_index = (node_community_before == community_index.view(-1, 1)).any(dim=0)
            new_index = ~true_index
            new_index_neighbor = neighbors_torch[new_index]
            node2community[new_index_neighbor] = index[new_index]


            #compare_index
            compare_index_neighbor = neighbors_torch[true_index]
            compare_community_before = node_community_before[true_index] #yuanlaide community
            community_member_before = community2node[compare_community_before]
            member_num_before = member_num[compare_community_before]
            avg_member_score_before_line = avg_member_score[compare_community_before]
            #avg_member_score_before = avg_member_score_before_line.gather(1,compare_index_neighbor.view(-1,1)).view(-1)
            compare_community_now = index[true_index]
            community_member_now = community2node[compare_community_now]
            member_num_now = member_num[compare_community_now]
            avg_member_score_now_line = avg_member_score[compare_community_now]
            for i in range(compare_index_neighbor.shape[0]):
                member_num_before_ = member_num_before[i]
                index_before = community_member_before[i][0:member_num_before_] == compare_index_neighbor[i]
                score_before = avg_member_score_before_line[i][0:member_num_before_][index_before]
                member_num_now_=member_num_now[i]
                #index_now = community_member_now[i][0:member_num_now_] == compare_index_neighbor[i]
                score_now = score[true_index][i] * member_num_now_
                #print(score_now)
                #print(score_before)
                #if(score_before.shape[0]==0):
                #    print("find")
                if score_before.shape[0]==0 or torch.ge(score_now,score_before)[0]:
                    node2community[compare_index_neighbor[i]] = compare_community_now[i]

            #avg_member_score_now = avg_member_score_now_line.gather(1, compare_index_neighbor.view(-1,1)).view(-1)

            #compare_result = torch.ge(avg_member_score_now,avg_member_score_before)
            #node2community[compare_index_neighbor][compare_result] = compare_community_now[compare_result]




        '''neighbors_new = neighbors_torch[new_index]
        new_index2 = index[new_index]
        node2community[neighbors_new] = new_index2  # TODO
        member_num[valid_nodes] = torch.tensor(neighbors_num).long().to(self.device)  # same length with valid nodes'''
        # print(member_num.dtype)
        # print(valid_nodes.dtype)
        if (community_layer > 1):
            other_index = ~new_index
            neighbors_other = neighbors_torch[other_index]
            node2community[neighbors_other] = node2community_layer[neighbors_other]

        del neighbors_torch
        if (community_layer > 1):
            return community_embeddings, community_score, community_index, member_score, node2community, community2node, member_num, \
                   community2node_layer, node2community_layer, member_num_layer
        else:
            return community_embeddings, community_score, community_index, member_score,avg_member_score, node2community, community2node, member_num, return_fit

    def forward2(self, memory, node_features, community_embeddings, unique_node_ids, timestamps, community_score,
                community_index,
                member_score, node2community, community2node, member_num, community_layer=1, community_percent=None,
                node2community_layer=None, community2node_layer=None, member_num_layer=None,
                n_neighbors=20, time_diffs=None, use_time_proj=True):

        neighbors, edge_idxs, edge_times, neighbors_unique, neighbors_num, valid_nodes = self.neighbor_finder.get_community_neighbor(
            unique_node_ids,
            timestamps,
            max_member_num=self.max_member_num,
            n_neighbors=n_neighbors)  # all_neighbor and itself

        neighbors_unique = np.array(neighbors_unique)

        neighbors_torch = torch.from_numpy(neighbors_unique).long().to(self.device)  # neighbors_unique
        neighbors_torch = neighbors_torch.reshape(-1)

        neighbors_features = node_features[neighbors_torch, :]  # unique neibor
        # gpu_tracker.track()

        if self.use_memory:
            memory_neighbors = memory.get_memory(neighbors_torch).detach()  # TODO tensor [neighbor_num, 1]
            neighbors_features = memory_neighbors + neighbors_features  # TODO

        index = []
        j = 0
        index1 = []
        for node in valid_nodes:
            for i in range(neighbors_num[j]):
                index.append(node)
                index1.append(j)
            j += 1
        # index = [node for node in valid_nodes for i in range(neighbors_num[i])] #every node's neighbor num not same
        # index is edge index 0
        # print(index)

        index = torch.tensor(index).long().to(self.device)
        index1 = torch.tensor(index1).long().to(self.device)

        valid_nodes = torch.tensor(valid_nodes).long().to(self.device)
        max_feature, _ = scatter_max(neighbors_features, index1, dim=0)  # (index.size(), feature_dim)
        max_feature_linear = self.linear_max(
            max_feature)  # TODO (valid_nodes.size(), feature_dim)  every node's cluster's feature
        # chongfu 10 bian
        # max_feature[valid_nodes]=max_feature
        max_feature_duplication = max_feature_linear[index1]
        center_feature_cat = torch.cat((max_feature_duplication, neighbors_features),
                                       dim=-1)  # TODO (neighbors_size, feature_dim*2)
        score = self.linear_score(center_feature_cat)  # calculate every member's lushudu of cluster
        # TODO neighbor chongfule every member of cluster'score of
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(src=score, index=index)  # TODO
        # score = score[index]
        score = F.dropout(score, p=self.dropout_att, training=self.training)

        score_feature = neighbors_features * score.view(-1, 1)  # 119，172 * 119，1
        cluster_feature = scatter_add(score_feature, index1, dim=0)  # (update node size , feature_dim)
        # cluster_feature = cluster_feature[index1]
        # community_embeddings[valid_nodes] = cluster_feature  # all nodes's community feature
        # neighbors_community_embeddings = community_embeddings[neighbors_unique] # TODO these feature didnot update  maybe this place can update every cluster's embedding
        # gpu_tracker.track()

        fitness = torch.sigmoid(self.community_score_calculator(community_embeddings, valid_nodes, index=index, index1=index1,
                                            neighbors_unique=neighbors_unique)).view(-1)  # score for every cluster
        # gpu_tracker.track()
        # score_cluster_featurev= torch.matmul(fitness.view(1,-1), cluster_feature)
        community_embeddings[valid_nodes] = cluster_feature
        return_fit = fitness
        # community_embeddings[valid_nodes] = torch.matmul(fitness.view(1,-1) , cluster_feature ) # all nodes's community feature

        for i in range(len(neighbors_num)):  # update member score
            j = neighbors_num[i]
            n = valid_nodes[i]
            neigh_index = index == n
            neigh = neighbors_torch[neigh_index]
            community2node[n][0:j] = neigh
            neigh_score = score[neigh_index].view(-1)
            member_score[n][0:j] = neigh_score
            member_num[n] = j

        if (community_layer > 1):
            new_center = valid_nodes
            center_member_num = neighbors_num  # first layer
            center_member_score = score
            layer_score = 0
            percent = community_percent[0]
            for j in range(len(valid_nodes)):  # loop for all valid nodes - one hop
                layer_member_num = int((center_member_num[j]) * percent)
                layer_top_score, layer_indices = member_score[new_center[j]][:center_member_num[j]].topk(
                    k=layer_member_num)
                layer_member = community2node[new_center[j]][layer_indices]  # one hop neighbor
                second_layer_member_all = community2node[layer_member]
                second_layer_score_all = member_score[layer_member]
                second_layer_num_all = member_num[layer_member]
                layer_score = 0
                add_community_member = []
                percent = community_percent[1]
                for z in range(len(layer_member)):  # loop for one hop neighbor - find 2 hop
                    if (layer_member[z] in add_community_member):
                        continue
                    second_layer_member_num = int((second_layer_num_all[z]) * percent)
                    second_layer_top_score, second_layer_indices = second_layer_score_all[z][
                                                                   :second_layer_num_all[z]].topk(
                        k=second_layer_member_num)
                    second_layer_member = second_layer_member_all[z][second_layer_indices]
                    second_layer_member_score = second_layer_score_all[z][second_layer_indices]
                    layer_score += torch.matmul(second_layer_member_score,
                                                layer_top_score[z].repeat(second_layer_member_num, 1))
                    add_community_member.extend(second_layer_member.tolist())
                    if (community_layer > 2):
                        third_layer_member_all = community2node[second_layer_member]
                        third_layer_score_all = member_score[second_layer_member]
                        third_layer_num_all = member_num[second_layer_member]
                        percent = community_percent[2]
                        for m in range(second_layer_member_num):
                            if (second_layer_member[m] in add_community_member):
                                continue
                            third_layer_member_num = int((third_layer_num_all[m]) * percent)
                            third_layer_top_score, third_layer_indices = third_layer_score_all[m][
                                                                         :third_layer_num_all[m]].topk(
                                k=third_layer_member_num)
                            third_layer_member = third_layer_member_all[m][third_layer_indices]
                            third_layer_member_score = third_layer_score_all[m][third_layer_indices]
                            layer_score += torch.matmul(third_layer_member_score,
                                                        second_layer_member_score[m].repeat(third_layer_member_num, 1))
                            add_community_member.extend(third_layer_member.tolist())
                fitness[j] = fitness[j] + layer_score
                # print(f'layer_score:{layer_score}')
                add_community_member = np.unique(np.array(add_community_member)).tolist()
                add_len = len(add_community_member)
                community2node_layer[new_center[j]][0:add_len] = torch.tensor(add_community_member).to(self.device)
                member_num_layer[new_center[j]] = add_len
                node2community_layer[add_community_member] = new_center[j]

        if (community_score.shape[0] != 0):
            fitness = torch.cat((fitness, community_score))
            community_index = torch.cat((valid_nodes, community_index))
            if (community_index.shape[0] < self.community_max_num):
                community_score = fitness
                new_community_index = community_index
            else:
                perm, indices = fitness.topk(k=self.community_max_num,
                                             largest=True)  # (x=fitness, ratio=self.ratio) # bao liu jie dian de bi li
                community_score = fitness[indices]  # perm is index
                new_community_index = torch.lt(indices, valid_nodes.shape[0])
                new_community_index = indices[new_community_index]
                new_community_index = community_index[new_community_index]
                community_index = community_index[indices]
        else:
            community_index = valid_nodes
            community_score = fitness
            new_community_index = community_index

        new_index = (index == new_community_index.view(-1, 1)).any(dim=0)
        neighbors_new = neighbors_torch[new_index]
        new_index2 = index[new_index]
        node2community[neighbors_new] = new_index2  # TODO
        member_num[valid_nodes] = torch.tensor(neighbors_num).long().to(self.device)  # same length with valid nodes

        if (community_layer > 1):
            other_index = ~new_index
            neighbors_other = neighbors_torch[other_index]
            node2community[neighbors_other] = node2community_layer[neighbors_other]

        del neighbors_torch

        if (community_layer > 1):
            return community_embeddings, community_score, community_index, member_score, node2community, community2node, member_num, \
                   community2node_layer, node2community_layer, member_num_layer
        else:
            return community_embeddings, community_score, community_index, member_score, node2community, community2node, member_num, return_fit

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__, self.in_channels, self.ratio)



    def forward4(self, memory, node_features, community_embeddings, unique_node_ids, timestamps, community_score,
                community_index,
                member_score, avg_member_score,
                 node2community, community2node, member_num, community_layer=1, community_percent=None,
                node2community_layer=None, community2node_layer=None, member_num_layer=None,
                # new_community_information=None,
                n_neighbors=20, time_diffs=None, use_time_proj=True):
    # for true community index
        # gpu_tracker.track()

        # nodes_torch = torch.from_numpy(nodes).long().to(self.device)  # update nodes
        # timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        # nodes_time_embedding = self.time_encoder(torch.zeros_like(
        #    timestamps_torch))

        # find max pool in every update node and their neighbors
        '''dynamic_new_center=[]
        dynamic_new_center_embeddings=[]
        dynamic_new_times = []'''
        neighbors, edge_idxs, edge_times, neighbors_unique, neighbors_num, valid_nodes = self.neighbor_finder.get_community_neighbor(
            unique_node_ids,
            timestamps,
            max_member_num=self.max_member_num,
            n_neighbors=n_neighbors)  # all_neighbor and itself

        neighbors_unique = np.array(neighbors_unique)

        neighbors_torch = torch.from_numpy(neighbors_unique).long().to(self.device)  # neighbors_unique
        neighbors_torch = neighbors_torch.reshape(-1)

        neighbors_features = node_features[neighbors_torch, :]  # unique neibor
        # gpu_tracker.track()

        if self.use_memory:
            memory_neighbors = memory.get_memory(neighbors_torch).detach()  # TODO tensor [neighbor_num, 1]
            neighbors_features = memory_neighbors + neighbors_features  # TODO

        index = []
        j = 0
        index1 = []
        for node in valid_nodes:
            for i in range(neighbors_num[j]):
                index.append(node)
                index1.append(j)
            j += 1
        # index = [node for node in valid_nodes for i in range(neighbors_num[i])] #every node's neighbor num not same
        # index is edge index 0
        # print(index)

        index = torch.tensor(index).long().to(self.device)
        index1 = torch.tensor(index1).long().to(self.device)

        valid_nodes = torch.tensor(valid_nodes).long().to(self.device)
        max_feature, _ = scatter_max(neighbors_features, index1, dim=0)  # (index.size(), feature_dim)
        max_feature_linear = self.linear_max(
            max_feature)  # TODO (valid_nodes.size(), feature_dim)  every node's cluster's feature
        # chongfu 10 bian
        # max_feature[valid_nodes]=max_feature
        max_feature_duplication = max_feature_linear[index1]
        center_feature_cat = torch.cat((max_feature_duplication, neighbors_features),
                                       dim=-1)  # TODO (neighbors_size, feature_dim*2)
        score = self.linear_score(center_feature_cat)  # calculate every member's lushudu of cluster
        # TODO neighbor chongfule every member of cluster'score of
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(src=score, index=index)  # TODO
        # score = score[index]
        score = F.dropout(score, p=self.dropout_att, training=self.training)

        score_feature = neighbors_features * score.view(-1, 1)  # 119，172 * 119，1
        cluster_feature = scatter_add(score_feature, index1, dim=0)  # (update node size , feature_dim)
        # cluster_feature = cluster_feature[index1]
        # community_embeddings[valid_nodes] = cluster_feature  # all nodes's community feature
        # neighbors_community_embeddings = community_embeddings[neighbors_unique] # TODO these feature didnot update  maybe this place can update every cluster's embedding
        # gpu_tracker.track()

        fitness = torch.sigmoid(
            self.community_score_calculator(community_embeddings, valid_nodes, index=index, index1=index1,
                                            neighbors_unique=neighbors_unique)).view(-1)  # score for every cluster
        # gpu_tracker.track()
        # score_cluster_featurev= torch.matmul(fitness.view(1,-1), cluster_feature)
        community_embeddings[valid_nodes] = cluster_feature
        return_fit = fitness
        # community_embeddings[valid_nodes] = torch.matmul(fitness.view(1,-1) , cluster_feature ) # all nodes's community feature

        for i in range(len(neighbors_num)):  # update member score
            j = neighbors_num[i]
            n = valid_nodes[i]
            #if n == 25507:
            #    print("find")
            neigh_index = index == n
            neigh = neighbors_torch[neigh_index]
            community2node[n][0:j] = neigh
            neigh_score = score[neigh_index].view(-1)
            member_score[n][0:j] = neigh_score
            avg_member_score[n][neigh] = neigh_score * j
            member_num[n] = j

        if (community_layer > 1):
            new_center = valid_nodes
            center_member_num = neighbors_num  # first layer
            center_member_score = score
            layer_score = 0
            percent = community_percent[0]
            for j in range(len(valid_nodes)):  # loop for all valid nodes - one hop
                layer_member_num = int((center_member_num[j]) * percent)
                layer_top_score, layer_indices = member_score[new_center[j]][:center_member_num[j]].topk(
                    k=layer_member_num)
                layer_member = community2node[new_center[j]][layer_indices]  # one hop neighbor
                second_layer_member_all = community2node[layer_member]
                second_layer_score_all = member_score[layer_member]
                second_layer_num_all = member_num[layer_member]
                layer_score = 0
                add_community_member = []
                percent = community_percent[1]
                for z in range(len(layer_member)):  # loop for one hop neighbor - find 2 hop
                    if (layer_member[z] in add_community_member):
                        continue
                    second_layer_member_num = int((second_layer_num_all[z]) * percent)
                    second_layer_top_score, second_layer_indices = second_layer_score_all[z][
                                                                   :second_layer_num_all[z]].topk(
                        k=second_layer_member_num)
                    second_layer_member = second_layer_member_all[z][second_layer_indices]
                    second_layer_member_score = second_layer_score_all[z][second_layer_indices]
                    layer_score += torch.matmul(second_layer_member_score,
                                                layer_top_score[z].repeat(second_layer_member_num, 1))
                    add_community_member.extend(second_layer_member.tolist())
                    if (community_layer > 2):
                        third_layer_member_all = community2node[second_layer_member]
                        third_layer_score_all = member_score[second_layer_member]
                        third_layer_num_all = member_num[second_layer_member]
                        percent = community_percent[2]
                        for m in range(second_layer_member_num):
                            if (second_layer_member[m] in add_community_member):
                                continue
                            third_layer_member_num = int((third_layer_num_all[m]) * percent)
                            third_layer_top_score, third_layer_indices = third_layer_score_all[m][
                                                                         :third_layer_num_all[m]].topk(
                                k=third_layer_member_num)
                            third_layer_member = third_layer_member_all[m][third_layer_indices]
                            third_layer_member_score = third_layer_score_all[m][third_layer_indices]
                            layer_score += torch.matmul(third_layer_member_score,
                                                        second_layer_member_score[m].repeat(third_layer_member_num, 1))
                            add_community_member.extend(third_layer_member.tolist())
                fitness[j] = fitness[j] + layer_score
                # print(f'layer_score:{layer_score}')
                add_community_member = np.unique(np.array(add_community_member)).tolist()
                add_len = len(add_community_member)
                community2node_layer[new_center[j]][0:add_len] = torch.tensor(add_community_member).to(self.device)
                member_num_layer[new_center[j]] = add_len
                node2community_layer[add_community_member] = new_center[j]

        # fitness = [fitness, community_score]
        # unique_node_ids = torch.tensor(unique_node_ids).to(self.device)
        if (community_score.shape[0] != 0):
            fitness = torch.cat((fitness, community_score))
            community_index = torch.cat((valid_nodes, community_index))
            unique_community_index,unique_indices = torch.unique(community_index,return_inverse=True)
            unique_fitness = torch.zeros(unique_community_index.shape[0]).to(self.device)
            unique_fitness[unique_indices] = fitness

            if (unique_community_index.shape[0] < self.community_max_num):
                community_score = unique_fitness
                community_index = unique_community_index
            else:
                perm, indices = unique_fitness.topk(k=self.community_max_num,
                                             largest=True)  # (x=fitness, ratio=self.ratio) # bao liu jie dian de bi li
                community_score = unique_fitness[indices]  # perm is index
                '''new_community_index = torch.lt(indices, valid_nodes.shape[0])
                new_community_index = indices[new_community_index]
                new_community_index = community_index[new_community_index]'''
                community_index = unique_community_index[indices]
        else:
            community_index = valid_nodes
            community_score = fitness
            #new_community_index = community_index
        # calculate jiao ji between perm and index
        if neighbors_torch.shape[0] != 0:
            node_community_before = node2community[neighbors_torch]

            true_index = (node_community_before == community_index.view(-1, 1)).any(dim=0)
            new_index = ~true_index
            new_index_neighbor = neighbors_torch[new_index]
            node2community[new_index_neighbor] = index[new_index]


            #compare_index
            compare_index_neighbor = neighbors_torch[true_index]
            compare_community_before = node_community_before[true_index] #yuanlaide community
            #community_member_before = community2node[compare_community_before]
            #member_num_before = member_num[compare_community_before]
            avg_member_score_before_line = avg_member_score[compare_community_before]
            avg_member_score_before = avg_member_score_before_line.gather(1,compare_index_neighbor.view(-1,1)).view(-1)
            compare_community_now = index[true_index]
            #community_member_now = community2node[compare_community_now]
            #member_num_now = member_num[compare_community_now]
            avg_member_score_now_line = avg_member_score[compare_community_now]


            avg_member_score_now = avg_member_score_now_line.gather(1, compare_index_neighbor.view(-1,1)).view(-1)

            compare_result = torch.ge(avg_member_score_now,avg_member_score_before)
            node2community[compare_index_neighbor][compare_result] = compare_community_now[compare_result]




        '''neighbors_new = neighbors_torch[new_index]
        new_index2 = index[new_index]
        node2community[neighbors_new] = new_index2  # TODO
        member_num[valid_nodes] = torch.tensor(neighbors_num).long().to(self.device)  # same length with valid nodes'''
        # print(member_num.dtype)
        # print(valid_nodes.dtype)
        if (community_layer > 1):
            other_index = ~new_index
            neighbors_other = neighbors_torch[other_index]
            node2community[neighbors_other] = node2community_layer[neighbors_other]

        del neighbors_torch
        if (community_layer > 1):
            return community_embeddings, community_score, community_index, member_score, node2community, community2node, member_num, \
                   community2node_layer, node2community_layer, member_num_layer
        else:
            return community_embeddings, community_score, community_index, member_score,avg_member_score, node2community, community2node, member_num, return_fit