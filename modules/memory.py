import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
                 device="cpu", combination_method='sum'):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.combination_method = combination_method
        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                                   requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                        requires_grad=False)
        self.nodes_connections = defaultdict(dict)

        self.messages = defaultdict(list)

    def add_connections(self, source_nodes, destination_nodes, times):
        instance_nums = source_nodes.shape[0]
        for i in range(instance_nums):
            s = source_nodes[i]
            d = destination_nodes[i]
            t = times[i]
            s_con = self.nodes_connections[s]
            if d not in s_con:
                self.nodes_connections[s][d] = [1,t]
            else:
                s_con[d][0] += 1
                s_con[d][1] = t

            d_con = self.nodes_connections[d]
            if s not in d_con:
                self.nodes_connections[d][s] = [1,t]
            else:
                d_con[s][0] += 1
                d_con[s][1] = t

    def get_connections(self, source_nodes, destination_nodes):
        instance_num = source_nodes.shape[0]

        s_d_nums = []
        s_d_times = []
        s_n_nums = []
        s_n_times = []
        for i in range(instance_num):
            s = source_nodes[i]
            d = destination_nodes[i]
            # n = nodes[i + 2 * instance_num]
            s_list = self.nodes_connections[s]
            if d not in s_list:
                s_d_nums.append(0)
                s_d_times.append(0)
            else:
                num, t=s_list[d]
                s_d_nums.append(num)
                s_d_times.append(t)

            # if n not in s_list:
            #     s_n_nums.append(0)
            #     s_n_times.append(0)
            # else:
            #     num, t = s_list[n]
            #     s_n_nums.append(num)
            #     s_n_times.append(t)
        return s_d_nums

    def clear_connections(self, source_nodes, destination_nodes):
        instance_num = source_nodes.shape[0]

        s_d_nums = []
        s_d_times = []
        s_n_nums = []
        s_n_times = []
        for i in range(instance_num):
            s = source_nodes[i]
            d = destination_nodes[i]
            # n = nodes[i + 2 * instance_num]
            s_list = self.nodes_connections[s]
            if d in s_list:
                s_list[d][0] = 0
            d_list = self.nodes_connections[d]
            if s in d_list:
                d_list[s][0] = 0

            # if n not in s_list:
            #     s_n_nums.append(0)
            #     s_n_times.append(0)
            # else:
            #     num, t = s_list[n]
            #     s_n_nums.append(num)
            #     s_n_times.append(t)
        return s_d_nums



    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach_()

        # Detach all stored messages
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []
