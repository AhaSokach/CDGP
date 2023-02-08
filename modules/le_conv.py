import torch
from torch.nn import Parameter
from torch_scatter import scatter_add



class LEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        
    def forward(self, all_community_embeddings, valid_nodes, index, index1, neighbors_unique, edge_weight=None, size=None):
        all_community_embeddings_weight = torch.matmul(all_community_embeddings, self.weight)

        if edge_weight is None:
            edge_weight = torch.ones((index1.size(0), ),
                                     dtype=all_community_embeddings.dtype,
                                     device=index1.device)
        deg = scatter_add(edge_weight, index1, dim=0)
        aggr_out = scatter_add(all_community_embeddings_weight[index], index1, dim=0)
        nodes_community_embeddings = all_community_embeddings[valid_nodes]
        lenv_score = (deg.view(-1, 1) * self.lin1(nodes_community_embeddings) + aggr_out) + self.lin2(nodes_community_embeddings) # different from expression [nodes, ]
        return lenv_score


def get_community_score_calculator(module_type, in_channels, out_channels):
    if module_type == "leconv":
        return LEConv(in_channels,out_channels)

