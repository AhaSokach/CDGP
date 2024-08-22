import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer, MLP_edge
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
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_nodes, n_edges,
                 dropout=0.1, community_updater_type="topk",
                 community_num=800, community_layer=1,
                 member_num=200,
                 # message_dimension=100,
                 memory_dimension=500, community_dimension=500,  # embedding_module_type="graph_attention",
                 message_function="identity",file = None,  aggregator_type="last",
                 memory_updater_type="gru",
                 ):
        super(CDGP, self).__init__()


        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        # if use_node_features:
        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        #  ...
        # if use_edge_features:
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device).reshape(-1, 1)
        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = n_nodes
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        # self.n_neighbors = n_neighbors
        # self.embedding_module_type = embedding_module_type

        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.community_num = community_num
        self.max_member_num = member_num
        self.memory = None

        self.community_dimension = community_dimension

        self.memory_dimension = memory_dimension

        message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                            self.time_encoder.dimension #  + 1

        # message_dimension = message_dimension if message_function != "identity" else raw_message_dimension

        self.memory = Memory(n_nodes=self.n_nodes,
                             memory_dimension=self.memory_dimension,
                             input_dimension=message_dimension,
                             message_dimension=message_dimension,
                             device=device)
        self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                         device=device)
        self.message_function = get_message_function(module_type=message_function,
                                                     raw_message_dimension=message_dimension,
                                                     message_dimension=message_dimension)
        self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                 memory=self.memory,
                                                 message_dimension=message_dimension,
                                                 memory_dimension=self.memory_dimension,
                                                 device=device)
        # print(self.memory_updater.memory_updater.weight_ih)
        # embedding_module_type = "graph_attention"
        # self.n_layers=1
        # self.embedding_module = get_embedding_module(module_type=embedding_module_type,
        #                                              node_features=self.node_raw_features,
        #                                              edge_features=self.edge_raw_features,
        #                                              memory=self.memory,
        #                                              neighbor_finder=self.neighbor_finder,
        #                                              time_encoder=self.time_encoder,
        #                                              n_layers=self.n_layers,
        #                                              n_node_features=self.n_node_features,
        #                                              n_edge_features=self.n_edge_features,
        #                                              n_time_features=self.n_node_features,
        #                                              embedding_dimension=self.embedding_dimension,
        #                                              device=self.device,
        #                                              n_heads=n_heads, dropout=dropout,
        #                                              use_memory="True",
        #                                              n_neighbors=self.n_neighbors)

        # self.community_memory_updater = self.memory_updater# .memory_updater
        self.community_memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                           memory=self.memory,
                                                           message_dimension=message_dimension,
                                                           memory_dimension=self.memory_dimension,
                                                           # init=True,
                                                           device=device)

        self.community_memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                           memory=self.memory,
                                                           message_dimension=message_dimension,
                                                           memory_dimension=self.memory_dimension,
                                                           device=device)

        self.community_updater_type = "topk"

        self.community_layer = community_layer
        self.community_percent = [1, 0.2, 0.1]
        self.file=file
        self.community = Community(n_nodes=self.n_nodes, community_num=self.community_num,
                                   max_member_num=self.max_member_num, community_layer=self.community_layer,
                                   community_percent=[1, 0.05],
                                   community_dimension=self.community_dimension,
                                   updator_type=self.community_updater_type,
                                   device=self.device)
        self.community_updater = get_community_updater(module_type=self.community_updater_type,
                                                       community_memory_updater=self.community_memory_updater,
                                                       community=self.community,
                                                       memory=self.memory,
                                                       community_dimension=self.community_dimension,
                                                       n_nodes=self.n_nodes,
                                                       community_num=self.community_num,
                                                       max_member_num=self.max_member_num,
                                                       community_layer=self.community_layer,
                                                       community_percent=self.community_percent,
                                                       score_type="community",
                                                       neighbor_finder=neighbor_finder,
                                                       dropout=dropout,
                                                       device=device)

        self.predictor_type = 'community'

        self.predictor = get_predictor(emb_dim=self.n_node_features, predictor_type=self.predictor_type)

        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)
        self.edge_predictor = MLP_edge(dim=self.memory_dimension * 4)

    def forward(self, source_nodes, destination_nodes, edge_times, edge_idxs, index):

        source_node_embedding, destination_node_embedding = self.compute_embeddings(
            source_nodes, destination_nodes, edge_times, edge_idxs)

        source_node_embedding = source_node_embedding[index]
        source_nodes = source_nodes[index]
        destination_nodes = destination_nodes[index]
        community_nodes = self.community.node2community[source_nodes]
        community_index = self.community.community_index
        use_index = (community_nodes.view(1, -1) == community_index.view(-1, 1)).any(dim=0)
        community_embedding = self.community.community_embeddings[source_nodes]
        pred = self.predictor.forward(source_node_embedding, community_embedding, use_index, self.device)
        self.memory.clear_connections(source_nodes,destination_nodes)

        return pred

    def forward_edge(self, source_nodes, destination_nodes, negative_nodes, event_times, event_ids):
        n_samples = len(source_nodes)
        source_node_memory = self.memory.get_memory(source_nodes)
        destination_node_memory = self.memory.get_memory(destination_nodes)
        negative_node_memory = self.memory.get_memory(negative_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        # community_nodes = self.community.node2community[nodes]
        # community_index = self.community.community_index
        # use_index = (community_nodes.view(1, -1) == community_index.view(-1, 1)).any(dim=0)
        community_embedding = self.community.community_embeddings[nodes]
        score_pos = self.edge_predictor(torch.cat([source_node_memory, community_embedding[:n_samples],
                                                      destination_node_memory,community_embedding[n_samples:2*n_samples]], dim=1))
        score_neg = self.edge_predictor(torch.cat([source_node_memory, community_embedding[:n_samples],
                                                      negative_node_memory,
                                                      community_embedding[n_samples*2:3 * n_samples]], dim=1))
        self.compute_embeddings(
            source_nodes, destination_nodes, event_times, event_ids)
        return score_pos, score_neg

    def compute_embeddings(self, source_nodes, destination_nodes, edge_times,
                           edge_idxs):

        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update



        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        message_nodes, messages, message_times = self.get_raw_messages(source_nodes,
                                                             source_node_embedding,
                                                             destination_nodes,
                                                             destination_node_embedding,
                                                             edge_times, edge_idxs)

        unique_nodes, unique_messages, unique_timestamps = self.update_memory(message_nodes, messages, message_times)
        self.update_community(unique_nodes, unique_messages, unique_timestamps)

        memory = self.memory.get_memory(list(range(self.n_nodes)))
        # node_embedding = self.embedding_module.compute_embedding(memory=memory,
        #                                                          source_nodes=nodes,
        #                                                          timestamps=timestamps,
        #                                                          n_layers=self.n_layers,
        #                                                          n_neighbors=n_neighbors,
        #                                                          time_diffs=time_diffs)
        #
        # source_node_embedding = node_embedding[:n_samples]
        # destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        # negative_node_embedding = node_embedding[2 * n_samples:]
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        # print(f's:{source_nodes}')
        # print(f'd:{destination_nodes}')
        # print(f's:{source_node_embedding}')
        # print(f'd:{destination_node_embedding}')
        # print(f's:{source_node_embedding.cpu().detach().numpy()}')
        # print(f'd:{destination_node_embedding.cpu().detach().numpy()}')
        return source_node_embedding, destination_node_embedding

    def update_memory(self, unique_nodes, unique_messages, unique_timestamps):
        # Aggregate messages for the same nodes
        # unique_nodes, unique_messages, unique_timestamps = \
        #     self.message_aggregator.aggregate(
        #         nodes,
        #         messages)
        #
        # if len(unique_nodes) > 0:
        #     unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)
        return unique_nodes, unique_messages, unique_timestamps

    def update_community(self, unique_nodes, unique_messages, unique_timestamps):
        self.community_updater.update_community(self.node_raw_features,
                                                unique_nodes, unique_timestamps)
        self.community_updater.update_community_member(unique_nodes,
                                                      unique_messages,
                                                    unique_timestamps,self.file)
        # self.embedding_module.update_memory(update_nodes, update_messages,
        #                                 timestamps = update_timestamps, change = False)

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
        edge_times2 = torch.cat((edge_times, edge_times))
        edge_features = self.edge_raw_features[edge_idxs]
        nodes = np.concatenate([source_nodes, destination_nodes])
        # self.memory.add_connections(source_nodes, destination_nodes, edge_times)
        # num = self.memory.get_connections(source_nodes, destination_nodes)
        # num = torch.tensor(num).to(self.device)
        source_memory = source_node_embedding
        destination_memory = destination_node_embedding

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
            source_nodes), -1)

        destination_time_delta = edge_times - self.memory.last_update[destination_nodes]
        destination_time_delta_encoding = self.time_encoder(destination_time_delta.unsqueeze(dim=1)).view(len(
            destination_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features, # num.unsqueeze(1),
                                    source_time_delta_encoding],
                                   dim=1)
        destination_message = torch.cat([destination_memory, source_memory, edge_features,#  num.unsqueeze(1),
                                         destination_time_delta_encoding], dim=1)

        messages_cat = torch.cat((source_message, destination_message), dim=0)
        # messages = defaultdict(list)
        # unique_nodes = np.unique(nodes)

        # for i in range(len(nodes)):
        #     messages[nodes[i]].append((messages_cat[i], edge_times2[i]))
        # unique_nodes, unique_messages, unique_timestamps = \
        #     self.message_aggregator.aggregate(
        #         nodes,
        #         messages)
        # print(f"for ")
        # print(unique_nodes)
        # print(unique_messages)
        # print(unique_timestamps)


        nodes_flip = np.flip(nodes)
        nodes_flip_unique, index_flip = np.unique(nodes_flip, return_index=True)
        index = (nodes_flip.shape[0] - 1) - index_flip
        messages_nodes = nodes[index]
        messages_times = edge_times2[index]
        # messages_cor_destination = destination_nodes[index]
        index_torch = torch.tensor(index).to(self.device)
        messages = messages_cat[index_torch]
        # print(messages_nodes)
        # print(messages)
        # print(messages_times)

        return messages_nodes, messages, messages_times

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
        self.community_updater.detector.neighbor_finder = neighbor_finder
