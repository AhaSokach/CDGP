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

    def forward(self, emb, community_emd,use_index,device):
        index = np.arange(0, emb.shape[0], 1)
        index = torch.from_numpy(index).to(device)
        pred = torch.zeros(emb.shape[0]).to(device)
        index1 = index[use_index]
        index2 = index[~use_index]
        emb1 = emb[index1]
        emb2 = emb[index2]
        community_emd = community_emd[index1]
        emb = torch.cat((emb1, community_emd),dim=1) # TODO emd require false
        pred1 = torch.squeeze(self.lins(emb), dim=1)
        pred2 = torch.squeeze(self.lins2(emb2), dim=1)
        pred[index1] =pred1
        pred[index2] = pred2
        return pred


def get_predictor(emb_dim, predictor_type='linear', merge_prob=0.5):
    if predictor_type == 'linear':
        return Linear(emb_dim)
    elif predictor_type == 'community':
        return LinearWithCommunity(emb_dim)

    else:
        raise ValueError('Not implemented predictor type!')


if __name__ == '__main__':
    ...
