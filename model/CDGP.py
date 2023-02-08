import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.community_updater import get_community_updater
from modules.embedding_module import get_embedding_module
from modules.community import Community
from model.time_encoding import TimeEncode
from modules.prediction import get_predictor



class CDGP(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device,n_nodes,n_edges, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False, use_community=True,community_updater_type="topk", community_num= 800,community_score_type="leconv",
                member_num=200,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, community_dimension = 500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
    super(CDGP, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)
    #if use_node_features:
    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    #  ...
    #if use_edge_features:
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device).reshape(-1,1)
    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = n_nodes
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.use_community = use_community
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.community_num = community_num
    self.max_member_num = member_num
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst
    self.community_dimension = memory_dimension

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    if self.use_community:
      self.community_updater_type = community_updater_type

      self.community_layer = 1
      self.community = Community(n_nodes=self.n_nodes, community_num = self.community_num,
                                 max_member_num = self.max_member_num,community_layer=self.community_layer,
                                 community_percent = [0.2,0.1,0.05],
                                 community_dimension = self.community_dimension,
                                 updator_type = self.community_updater_type,
                                 device = self.device)
      self.community_updater = get_community_updater(module_type=self.community_updater_type,
                                                     community = self.community,
                                                     memory = self.memory,
                                                     community_dimension=self.community_dimension,
                                                     n_nodes=self.n_nodes,
                                                     community_num=self.community_num,
                                                     max_member_num = self.max_member_num,
                                                     score_type = community_score_type,
                                                     neighbor_finder=neighbor_finder,
                                                     dropout = dropout,
                                                     device=device)


    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    if(self.use_community and self.community_updater_type == "topk"):
      self.predictor_type='community'
    else:
      self.predictor_type = 'linear'
    self.predictor = get_predictor(emb_dim=self.n_node_features, predictor_type=self.predictor_type)

    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, gpu_tracker, n_neighbors=20):


    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    positives = nodes
    timestamps = np.concatenate([edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      source_time_diffs = torch.from_numpy(edge_times).long().to(self.device) - last_update[
        source_nodes].long() #
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs],
                             dim=0)

    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

    if self.use_memory:
      if self.memory_update_at_start:
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages,gpu_tracker)
        self.update_memory(unique_destinations, destination_id_to_messages,gpu_tracker)
      #gpu_tracker.track()



      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
    #print("source_node_embedding in em be return" + str(source_node_embedding.shape))
    return source_node_embedding, destination_node_embedding


  def compute_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, gpu_tracker, n_neighbors=20):

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    if self.use_community and self.community_updater_type == "kmeans":
        self.community.add_node(nodes)
    positives = nodes
    timestamps = np.concatenate([edge_times, edge_times])
    memory = self.memory.get_memory(list(range(self.n_nodes)))
    source_node_embedding = memory[source_nodes]
    destination_node_embedding = memory[destination_nodes]
    unique_nodes, id_to_messages = self.get_raw_messages(source_nodes,
                                                                  source_node_embedding,
                                                                  destination_nodes,
                                                                  destination_node_embedding,
                                                                  edge_times, edge_idxs)

    return_fit = self.update_memory(unique_nodes, id_to_messages, gpu_tracker)
    last_update = self.memory.last_update
    source_time_diffs = torch.from_numpy(edge_times).to(self.device) - last_update[
      source_nodes] #
    source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
    destination_time_diffs = torch.from_numpy(edge_times).to(self.device) - last_update[
      destination_nodes]
    destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst


    time_diffs = torch.cat([source_time_diffs, destination_time_diffs],
                             dim=0)
    memory = self.memory.get_memory(list(range(self.n_nodes)))
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    # Compute the embeddings using the embedding module

    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

    return source_node_embedding, destination_node_embedding,return_fit

  def forward(self, source_nodes, destination_nodes, edge_times,
                                 edge_idxs, index, n_neighbors=20):
      n_samples = len(source_nodes)
      source_node_embedding, destination_node_embedding,return_fit = self.compute_embeddings(
          source_nodes, destination_nodes, edge_times, edge_idxs,  n_neighbors)

      source_node_embedding = source_node_embedding[index]
      source_nodes = source_nodes[index]

      if(self.predictor_type == "community"):
        community_nodes = self.community.node2community[source_nodes]
        community_index = self.community.community_index
        use_index = (community_nodes.view(1, -1) == community_index.view(-1, 1)).any(dim=0)
        community_embedding = self.community.community_embeddings[source_nodes]
        pred = self.predictor.forward(source_node_embedding,community_embedding,use_index,self.device)
      elif(self.predictor_type == "community_weight"):
        pred = self.predictor.forward(source_node_embedding,self.community.node2community,
                                      self.community.community2node,self.community.member_score,
                                      self.community.member_num,
                                      self.community.community_embeddings,
                                      self.community.community_index,
                                      source_nodes,self.device)
      elif(self.predictor_type == "kmeans"):
          pred = self.predictor.forward(source_node_embedding, self.community.node2community,
                                        self.community.community_embeddings,
                                        source_nodes, self.device)
      else:
        pred = self.predictor.forward(source_node_embedding)
      return pred,return_fit

  def update_memory(self, nodes, messages, gpu_tracker):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

    if self.use_community and self.community_updater_type == "topk":
        return_fit = self.community_updater.update_community(self.node_raw_features,
                                                unique_nodes, unique_timestamps,gpu_tracker)

    else:
        return_fit=1
    return return_fit

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(  # mean or last
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_times2 = torch.cat((edge_times,edge_times))
    edge_features = self.edge_raw_features[edge_idxs]
    nodes = np.concatenate([source_nodes, destination_nodes])

    source_memory = source_node_embedding
    destination_memory = destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    destination_time_delta = edge_times - self.memory.last_update[destination_nodes]
    destination_time_delta_encoding = self.time_encoder(destination_time_delta.unsqueeze(dim=1)).view(len(
        destination_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    destination_message = torch.cat([destination_memory, source_memory, edge_features,
                                     destination_time_delta_encoding],dim=1)
    messages_cat = torch.cat((source_message, destination_message), dim=0)
    messages = defaultdict(list)
    unique_nodes = np.unique(nodes)

    for i in range(len(nodes)):
      messages[nodes[i]].append((messages_cat[i], edge_times2[i]))

    return unique_nodes, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
    if(self.use_community and self.community_updater_type=="topk"):
      self.community_updater.pooling.neighbor_finder = neighbor_finder

