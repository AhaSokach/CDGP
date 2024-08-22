import torch.nn as nn
import torch
import numpy as np


class Linear(nn.Module):
    def __init__(self, emb_dim):
        super(Linear, self).__init__()
        self.lins = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                  nn.ReLU(), nn.Linear(emb_dim // 2, 1))

    def forward(self, emb):
        return torch.squeeze(self.lins(emb), dim=1)


class LinearWithCommunity(nn.Module):
    def __init__(self, emb_dim):
        super(LinearWithCommunity, self).__init__()
        self.lins = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim // 2),
                                  nn.ReLU(), nn.Linear(emb_dim // 2, 1))
        self.lins2 = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                   nn.ReLU(), nn.Linear(emb_dim // 2, 1))

    def forward(self, emb, community_emd, use_index, device):
        index = np.arange(0, emb.shape[0], 1)
        index = torch.from_numpy(index).to(device)
        pred = torch.zeros(emb.shape[0]).to(device)
        index1 = index[use_index]
        index2 = index[~use_index]
        emb1 = emb[index1]
        emb2 = emb[index2]
        community_emd = community_emd[index1]
        emb = torch.cat((emb1, community_emd), dim=1)  # TODO emd require false
        pred1 = torch.squeeze(self.lins(emb), dim=1)
        pred2 = torch.squeeze(self.lins2(emb2), dim=1)
        pred[index1] = pred1
        pred[index2] = pred2
        return pred

    def forward2(self, emb, community_emd, use_index, device):
        emb = torch.cat((emb, community_emd), dim=1)  # TODO emd require false
        pred = torch.squeeze(self.lins(emb), dim=1)
        return pred


    def forward3(self, emb, community_emd, use_index, device):
        # emb = torch.cat((emb, community_emd), dim=1)  # TODO emd require false
        pred = torch.squeeze(self.lins2(emb), dim=1)
        return pred

class AttentionWithCommunity(nn.Module):
    def __init__(self, emb_dim):
        super(AttentionWithCommunity, self).__init__()
        self.lins = nn.Sequential(nn.Linear(emb_dim * 3, emb_dim),
                                  nn.ReLU(), nn.Linear(emb_dim, 1))
        self.lins2 = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                   nn.ReLU(), nn.Linear(emb_dim // 2, 1))

    def forward(self, node_emb, node2community, community2node, member_score, member_num, community_embeddings,
                community_index, nodes, device):
        index = np.arange(0, node_emb.shape[0], 1)
        index = torch.from_numpy(index).to(device)
        pred = torch.zeros(node_emb.shape[0]).to(device)
        community_nodes = node2community[nodes]  # nodes' community
        use_index = (community_nodes.view(1, -1) == community_index.view(-1, 1)).any(dim=0)  # nodes's use community
        community_embeddings_nodes = community_embeddings[nodes]
        index1 = index[use_index]
        index2 = index[~use_index]
        emb1 = node_emb[index1]
        emb2 = node_emb[index2]
        community_emd = community_embeddings_nodes[index1]  # nodes'

        use_community_nodes = community_nodes[index1]
        # use community's nodes
        community_nodes = community2node[use_community_nodes]
        nodes_score = member_score[use_community_nodes]
        nodes_member_num = member_num[use_community_nodes]
        member_embedding = torch.zeros((emb1.shape[0], node_emb.shape[1])).to(device)

        for i in range(emb1.shape[0]):
            num = nodes_member_num[i]
            neighbors = community_nodes[i][0:num]
            neighbors_com = node2community[neighbors]
            use_neighbor_index = (neighbors_com.view(1, -1) == community_index.view(-1, 1)).any(dim=0)
            use_neighbor_com = neighbors_com[use_neighbor_index]
            neighbors_com_emb = community_embeddings[use_neighbor_com]
            score = nodes_score[i][0:num][use_neighbor_index]
            emb = torch.matmul(score.view(1, -1), neighbors_com_emb)
            member_embedding[i] = emb

        emb = torch.cat((emb1, community_emd, member_embedding), dim=1)  # TODO emd require false

        pred1 = torch.squeeze(self.lins(emb), dim=1)
        pred2 = torch.squeeze(self.lins2(emb2), dim=1)
        pred[index1] = pred1
        pred[index2] = pred2

        '''
        node_community = node2community[nodes]
        community_nodes = community2node[node_community]
        nodes_score = member_score[node_community]
        nodes_member_num = member_num[node_community]
        member_embedding = torch.zeros((node_emb.shape[0],node_emb.shape[1])).to(device)
        for i in range(community_nodes.shape[0]):
            num = nodes_member_num[i]
            comms = community_nodes[i][0:num]
            comms_emb = community_embedding[comms]
            score = nodes_score[i][0:num]
            emb = torch.matmul(score.view(1,-1), comms_emb)
            member_embedding[i] = emb
            #comms_unique = np.unique(comms)
        com_emb = community_embedding[node_community]
        embedding = torch.cat((node_emb,com_emb), dim=1)
        pred = torch.squeeze(self.lins(embedding), dim=1)'''
        return pred

    def forward2(self, emb, community_emd, use_index, device):
        emb = torch.cat((emb, community_emd), dim=1)  # TODO emd require false
        pred = torch.squeeze(self.lins(emb), dim=1)
        return pred


class MergeLinear(nn.Module):
    def __init__(self, emb_dim, prob):
        super(MergeLinear, self).__init__()
        self.prob = nn.Parameter(torch.tensor(prob), requires_grad=False)
        self.dynamic_fn = Linear(emb_dim)
        self.static_fn = Linear(emb_dim)

    def forward(self, emb):
        static_emb, dynamic_emb = emb
        pred = self.prob * self.static_fn(static_emb) + (1 - self.prob) * self.dynamic_fn(dynamic_emb)
        return pred


def get_predictor(emb_dim, predictor_type='linear', merge_prob=0.5):
    if predictor_type == 'linear':
        return Linear(emb_dim)
    elif predictor_type == 'merge':
        return MergeLinear(emb_dim, merge_prob)
    elif predictor_type == 'community':
        return LinearWithCommunity(emb_dim)
    elif predictor_type == 'community_weight':
        return AttentionWithCommunity(emb_dim)
    else:
        raise ValueError('Not implemented predictor type!')


if __name__ == '__main__':
    ...
