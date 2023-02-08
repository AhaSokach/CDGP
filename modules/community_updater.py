from torch import nn
import torch
from modules.le_conv import LEConv
from modules.asap_pool import Pooling
import numpy as np
from sklearn.cluster import KMeans
import torch_scatter
from torch_scatter import scatter_add, scatter_max

class CommunityUpdater(nn.Module):
  def update_community(self, unique_node_ids, unique_messages, timestamps):
    pass


class TopkCommunityUpdater(CommunityUpdater):
  def __init__(self, community, memory, community_dimension, n_nodes, community_max_num,max_member_num, score_type, neighbor_finder,dropout, device):
    super(TopkCommunityUpdater, self).__init__()
    self.community = community
    self.memory = memory
    self.community_dimension = community_dimension
    self.node2community = np.array([-1] * n_nodes)
    self.community2node = np.array([])
    self.community_score = [] #
    self.community_index = []
    self.community_max_num = community_max_num
    self.pooling = Pooling(community_dimension, neighbor_finder, community_max_num,max_member_num,score_type, device, dropout_att=dropout)
    self.device = device

  def update_community(self, node_features,unique_node_ids, timestamps):
    # node2community, community2node, member_score = self.community.get_community_relationship()
    community_embeddings, community_score, community_index = self.community.get_community_information()
    node2community, community2node, member_score,avg_member_score, member_num = self.community.get_community_relationship()
    if (self.community.community_layer > 1):
      community_embeddings,community_score, community_index,member_score, node2community, community2node, member_num, \
      community2node_layer, node2community_layer, member_num_layer \
        = \
        self.pooling.forward(self.memory, node_features,
                              community_embeddings, unique_node_ids,  timestamps,
                              community_score,community_index,
                             member_score, node2community, community2node, member_num,
                             self.community.community_layer,self.community.community_percent,
                  self.community.node2community_layer, self.community.community2node_layer, self.community.member_num_layer)
      self.community.set_layer(community2node_layer, node2community_layer, member_num_layer)
      return_fit = 1
    else:
      community_embeddings, community_score, community_index, member_score,avg_member_score, node2community, community2node, member_num ,return_fit\
        = \
        self.pooling.forward(self.memory, node_features,
                             community_embeddings, unique_node_ids, timestamps,
                             community_score, community_index,
                             member_score, avg_member_score, node2community, community2node, member_num)
    self.community.set_community_information(community_embeddings,community_score, community_index)
    self.community.set_community_relationship(node2community, community2node, member_score, avg_member_score, member_num)
    return return_fit


  def update_community_member(self,nodes,unique_message,timestamps):
    node2community, community2node, member_score, member_num = self.community.get_community_relationship()
    community_embeddings, community_score, community_index = self.community.get_community_information()
    community_nodes = node2community[nodes]
    use_index = (community_nodes.view(1, -1) == community_index.view(-1, 1)).any(dim=0)
    use_community = community_nodes[use_index]
    use_timestamps = timestamps[use_index]
    use_messages = unique_message[use_index]
    update_nodes = torch.tensor([]).to(self.device)
    update_messages = torch.tensor([]).to(self.device)
    update_timestamps = torch.tensor([]).to(self.device)
    for i in range (use_community.shape[0]):
      cid = use_community[i] # community id
      community_member = community2node[cid] # community member (1, max member)
      member_num_cid = member_num[cid].item()
      member_score_cid = member_score[cid]
      member_score_cid = member_score_cid[0:member_num_cid]
      messages = torch.mm(member_score_cid.view(-1,1), use_messages[i].view(1,-1))
      update_messages = torch.cat((update_messages,messages),dim=0)
      update_nodes = torch.cat((update_nodes, community_member[0:member_num_cid]))
      update_timestamps = torch.cat((update_timestamps,
                                     torch.repeat_interleave(use_timestamps[i],member_num_cid)))
    update_nodes = update_nodes.cpu().numpy()
    return update_nodes, update_messages,update_timestamps





def get_community_updater(module_type, community, memory,  community_dimension, n_nodes, community_num, max_member_num,score_type,neighbor_finder, dropout,device):
  if module_type == "topk":
    return TopkCommunityUpdater(community, memory,  community_dimension, n_nodes, community_num, max_member_num,score_type, neighbor_finder, dropout,device)



