from torch import nn
import torch
from modules.community_score_calculator import CommunityCalculator
from modules.community_detector import CommunityDetector
import numpy as np


class CommunityUpdater(nn.Module):
    def update_community(self, unique_node_ids, unique_messages, timestamps):
        pass


class TopkCommunityUpdater(CommunityUpdater):
    def __init__(self, community, community_memory_updater, memory, community_dimension, n_nodes, community_max_num,
                 max_member_num, community_layer,
                 community_percent, score_type, neighbor_finder, dropout, device):
        super(TopkCommunityUpdater, self).__init__()
        self.community = community
        self.memory_updater = community_memory_updater
        self.memory = memory
        self.community_dimension = community_dimension
        self.node2community = np.array([-1] * n_nodes)
        self.community2node = np.array([])
        self.community_score = []  #
        self.community_index = []
        self.community_max_num = community_max_num
        self.community_layer = community_layer
        self.community_percent = community_percent
        # self.gnn_score = LEConv(community_dimension,1)
        self.detector = CommunityDetector(community_dimension, neighbor_finder, community_max_num, max_member_num,
                                          score_type, device, dropout_att=dropout)
        self.device = device

    def update_community(self, node_features, unique_node_ids, timestamps):
        # node2community, community2node, member_score = self.community.get_community_relationship()
        community_embeddings, community_score, community_index = self.community.get_community_information()
        node2community, community2node, member_score, member_num = self.community.get_community_relationship()

        if self.community_layer == 1:
            community_embeddings, community_score, community_index, member_score, node2community, community2node, member_num \
                = \
                self.detector.forward(self.memory, node_features,
                                      community_embeddings, unique_node_ids, timestamps,
                                      community_score, community_index,
                                      member_score, node2community, community2node, member_num,
                                      self.community_layer, self.community_percent)
        else:
            community2link, link_score, link_num = self.community.get_link_relationship()
            (community_embeddings, community_score, community_index, member_score, node2community, community2node,
             member_num, community2link, link_score, link_num)\
                = \
                self.detector.forward(self.memory, node_features,
                                      community_embeddings, unique_node_ids, timestamps,
                                      community_score, community_index,
                                      member_score, node2community, community2node, member_num,
                                      self.community_layer, self.community_percent,
                                      community2link, link_score, link_num)
            self.community.set_link_relationship(community2link, link_score, link_num)

        self.community.set_community_information(community_embeddings, community_score, community_index)
        self.community.set_community_relationship(node2community, community2node, member_score, member_num)

    def update_community_member(self, nodes, unique_message, timestamps, file, threshold=0.3, para = 0.01):
        node2community, community2node, member_score, member_num = self.community.get_community_relationship()
        community_embeddings, community_score, community_index = self.community.get_community_information()
        community_nodes = torch.tensor(nodes).to(self.device)  # node2community[nodes] TODO _member
        # print(community_nodes)
        use_index = (community_nodes.view(1, -1) == community_index.view(-1, 1)).any(dim=0)
        use_community = community_nodes[use_index]
        use_timestamps = timestamps[use_index]
        use_messages = unique_message[use_index]

        update_nodes = torch.tensor([]).to(self.device)
        update_messages = torch.tensor([]).to(self.device)
        update_timestamps = torch.tensor([]).to(self.device)

        if use_community.shape[0]>0:

            community_member = community2node[use_community]  # community member (1, max member)
            member_num_cid = member_num[use_community]
            member_score_cid = member_score[use_community]
            use_member_score = torch.ge(member_score_cid, threshold)
            member_sum = torch.sum(use_member_score.int(), dim=1)
            # print(member_sum.cpu().numpy())
            member_sum_max = torch.max(member_num_cid)
            use_member_score_max = use_member_score[:,:member_sum_max]
            # member_score_batch =
            messages_dup = use_messages.repeat(1, member_sum_max).view(member_num_cid.shape[0],-1,use_messages.shape[1])

            # messages_dup = torch.repeat_interleave(use_messages, (member_num_cid.shape[0], member_sum_max))
            # messages = torch.mm(member_score_cid[:,:member_sum_max][use_member_score_max].view(-1, 1), messages_dup[use_member_score_max].view(1, -1))
            update_messages = torch.mul(member_score_cid[:,:member_sum_max][use_member_score_max].view(-1, 1), messages_dup[use_member_score_max])
            # print(update_messages.shape)
            update_nodes = community_member[:,:member_sum_max][use_member_score_max]
            # print(update_nodes.shape)
            update_timestamps = use_timestamps.view(-1,1).repeat(1, member_sum_max)[use_member_score_max]
        # update_timestamps = time_dup


        # for i in range(use_community.shape[0]):
        #     cid = use_community[i]  # community id
        #     community_member = community2node[cid]  # community member (1, max member)
        #     member_num_cid = member_num[cid].item()
        #     member_score_cid = member_score[cid]
        #     use_member_score_cid = torch.ge(member_score_cid, threshold)
        #
        #     member_score_cid = member_score_cid[use_member_score_cid][:member_num_cid] # * para
        #
        #     messages = torch.mm(member_score_cid.view(-1, 1), use_messages[i].view(1, -1))
        #     update_messages = torch.cat((update_messages, messages), dim=0)
        #
        #     update_nodes = torch.cat((update_nodes, community_member[use_member_score_cid][:member_num_cid]))
        #
        #     update_timestamps = torch.cat((update_timestamps,
        #                                    torch.repeat_interleave(use_timestamps[i], member_num_cid)))

        # update_message = member_score.view(1,-1) * unique_message  # TODO [all_member, feature dim]
        # update_timestamps = timestamps # TODO
        update_nodes = update_nodes.cpu().numpy()
        self.memory_updater.update_memory_member(update_nodes, update_messages,file=file,
                                         timestamps=update_timestamps, change=False)

        return update_nodes, update_messages, update_timestamps


def get_community_updater(module_type, community_memory_updater, community, memory, community_dimension, n_nodes,
                          community_num, max_member_num, community_layer, community_percent, score_type,
                          neighbor_finder, dropout, device):
    if module_type == "topk":
        return TopkCommunityUpdater(community, community_memory_updater, memory, community_dimension, n_nodes,
                                    community_num, max_member_num, community_layer, community_percent, score_type,
                                    neighbor_finder, dropout, device)
