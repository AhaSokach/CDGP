from torch import nn
import torch


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass

    def __init_weight__(self):
        self.lins.weight.data = torch.zeros_like(self.lins.weight.data)


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        emb_dim = message_dimension
        self.lins = nn.Linear(message_dimension + memory_dimension, memory_dimension, bias=False) #nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                     #             nn.ReLU(), nn.Linear(emb_dim // 2, 1))
        self.lins.weight.data = torch.zeros_like(self.lins.weight.data)
        self.linear2 = nn.Linear(message_dimension, memory_dimension, bias=False)
        self.linear2.weight.data = torch.zeros_like(self.linear2.weight.data)
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps, change=True):
        if len(unique_node_ids) <= 0:
            return

        memory = self.memory.get_memory(unique_node_ids)
        if change:
            self.memory.last_update[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)
        # print(f'd:{self.memory_updater.bias_hh.cpu().detach().numpy()}')

        self.memory.set_memory(unique_node_ids, updated_memory)


    def update_memory_member(self, unique_node_ids, unique_messages, file, timestamps, para = 0.0, change=True):
        if len(unique_node_ids) <= 0:
            return

        memory = self.memory.get_memory(unique_node_ids).data.clone()
        if change:
            self.memory.last_update[unique_node_ids] = timestamps

        weight = self.lins(torch.cat((unique_messages, memory),dim=1))
        weight = torch.tanh(weight)
        weight = torch.relu(weight) * para

        # print(weight.detach().cpu().numpy(),file=file)
        updated_memory = memory * (1 - weight) + weight * torch.tanh(self.linear2(unique_messages))
        # updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)



    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)



def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
