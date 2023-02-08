import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy
import numpy as np


class Community(nn.Module):

  def __init__(self, n_nodes, community_num,max_member_num,community_dimension,community_layer,community_percent,
               updator_type = "kmeans" ,device="cpu"):
    super(Community, self).__init__()
    self.n_nodes = n_nodes
    self.community_num = community_num
    self.max_member_num = max_member_num
    self.community_dimension = community_dimension
    self.community_layer = community_layer
    self.community_percent = community_percent
    self.device = device
    self.updator_module =updator_type

    self.__init_community__()

  def __init_community__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model


    self.community_embeddings = torch.zeros((self.n_nodes, self.community_dimension)).to(self.device)
                                # not all embedding is usefull

    self.node2community = torch.tensor([-1] * self.n_nodes).to(self.device)
    self.community2node = torch.zeros(self.n_nodes, self.max_member_num, dtype=int).to(self.device)
    self.community_score = torch.tensor([]).to(self.device)
    self.member_score = torch.zeros(self.n_nodes, self.max_member_num).to(self.device)
    self.avg_member_score = torch.zeros(self.max_member_num, self.max_member_num).to(self.device)
    self.member_num = torch.zeros(self.n_nodes, dtype=int).to(self.device)
    self.community_index = torch.tensor([]).to(self.device)
    if(self.updator_module == "kmeans"):
      self.evolving_node = np.array([])
      self.node2community = torch.tensor([-1] * self.n_nodes , dtype=int).to(self.device)

    if(self.community_layer>1):
      self.node2community_layer = torch.zeros(self.n_nodes, dtype=int).to(self.device)
      self.community2node_layer = torch.zeros(self.n_nodes, self.max_member_num, dtype=int).to(self.device)
      self.member_num_layer = torch.zeros(self.n_nodes, dtype=int).to(self.device)

  def get_community_information(self):
    return self.community_embeddings, self.community_score, self.community_index 

  def get_community_information_kmeans(self):
    return self.community_embeddings, self.node2community, self.evolving_node

  def set_community_information_kmeans(self, node2community, community_embeddings):
    self.node2community = node2community
    self.community_embeddings = community_embeddings

  def get_community_relationship(self):
    return self.node2community, self.community2node, self.member_score, self.avg_member_score,self.member_num

  def set_community_relationship(self, node2community, community2node, member_score,avg_member_score, member_num):
    self.node2community = node2community
    self.community2node = community2node
    self.member_score = member_score
    self.member_num = member_num
    self.avg_member_score = avg_member_score

  def set_community_information(self, community_embeddings, community_score, community_index):

    self.community_embeddings = community_embeddings
    self.community_score = community_score
    self.community_index = community_index

  def set_layer(self,community2node_layer, node2community_layer, member_num_layer):
    self.node2community_layer = node2community_layer
    self.community2node_layer = community2node_layer
    self.member_num_layer = member_num_layer

  def detach_community(self):
    if self.updator_module == "topk":
      self.member_score.detach_()
      self.community_score.detach_()
      self.community_embeddings.detach_()
      self.avg_member_score.detach_()


  def detach_community2(self):
    if self.updator_module == "topk":
      self.member_score = self.member_score.detach()
      self.community_score = self.community_score.detach()
      self.community_embeddings = self.community_embeddings.detach()
      self.avg_member_score = self.avg_member_score.detach()

  def add_node(self, nodes):
    evolving_node = np.concatenate([self.evolving_node,nodes])
    evolving_node = np.unique(evolving_node)
    self.evolving_node = evolving_node