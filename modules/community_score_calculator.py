import torch
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_add

from torch_geometric.nn.inits import uniform


class CommunityCalculator(torch.nn.Module):
    """
    Given embeddings of communities, calculate the community scores
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(CommunityCalculator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.weight = torch.nn.Linear(in_channels, out_channels, bias=bias) #Parameter(torch.Tensor(memory_dimension, memory_dimension))

    def calculate(self, all_community_embeddings, memory, valid_nodes, index, index1, neighbors_unique,\
                  index_noself, index1_noself, neighbors_unique_noself ,edge_weight=None,
                  size=None):
        """"""  # valid

        all_community_embeddings_weight = self.weight(all_community_embeddings)
            # torch.matmul(all_community_embeddings,
            #                                           self.weight))

        if edge_weight is None:
            edge_weight = torch.ones((index1.size(0),),  # for every edge a weight
                                     dtype=all_community_embeddings.dtype,
                                     device=index1.device)
        deg = scatter_add(edge_weight, index1, dim=0)  # + 1e-10
        aggr_out = scatter_add(edge_weight.view(-1, 1) * all_community_embeddings_weight[neighbors_unique], index1,
                               dim=0)
        nodes_community_embeddings = all_community_embeddings[valid_nodes]
        lenv_score = (deg.view(-1, 1) * self.lin1(nodes_community_embeddings) + aggr_out) + self.lin2(
            nodes_community_embeddings)
        return lenv_score

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def get_community_score_calculator(module_type, in_channels, out_channels):
    if module_type == "community":
        return CommunityCalculator(in_channels, out_channels)
