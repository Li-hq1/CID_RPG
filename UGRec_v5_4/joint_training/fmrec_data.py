"""FMRec prefix expansion: one last-item target, no target inside history."""

import numpy as np
import torch
from torch.utils.data import Dataset


class PrefixDataset(Dataset):
    def __init__(self,sequences,max_len):
        self.sequences = sequences
        self.max_len = max_len
        self.counts = np.cumsum([max(0,len(seq)-1) for seq in sequences])
        if not len(self.counts) or self.counts[-1]<1:raise ValueError('No training prefixes.')

    def __len__(self):return int(self.counts[-1])

    def __getitem__(self,index):
        user = int(np.searchsorted(self.counts,index,side='right'))
        offset = index-(int(self.counts[user-1]) if user else 0)+1
        seq = self.sequences[user]
        history = seq[max(0,offset-self.max_len):offset]
        ids = torch.zeros(self.max_len,dtype=torch.long)
        ids[-len(history):] = torch.tensor(history)
        return ids,torch.tensor(seq[offset])


def left_padded_batch(histories,indices,max_len):
    ids = torch.zeros(len(indices),max_len,dtype=torch.long)
    for row,index in enumerate(indices):
        history = histories[index][-max_len:]
        if not history:raise ValueError('Empty history.')
        ids[row,-len(history):] = torch.tensor(history)
    return ids
