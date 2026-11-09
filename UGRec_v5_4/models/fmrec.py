"""FMRec core ported from FengLiu-1/FMRec commit 1c39b17e59a08980c8372e4f861167e0a259c349.

Preserves reference query-mask attention, no positional embeddings, stochastic
token fusion (also in eval), endpoint prediction, and fixed-z0 Euler drift.
Only unused layers/imports and an optional history decoder are omitted.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        return self.weight * (x-mean) / torch.sqrt((x-mean).square().mean(-1, keepdim=True)+self.variance_epsilon) + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        for layer in [self.w_1, self.w_2]:
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = .5*hidden*(1+torch.tanh(math.sqrt(2/math.pi)*(hidden+.044715*hidden.pow(3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        self.size_head, self.num_heads = hidden_size//heads, heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch = len(q)
        q,k,v = [layer(x).view(batch,-1,self.num_heads,self.size_head).transpose(1,2)
                 for layer,x in zip(self.linear_layers,(q,k,v))]
        corr = q @ k.transpose(-2,-1) / math.sqrt(q.size(-1))
        if mask is not None:
            # Faithful to the repository: masks query rows, NOT padding keys.
            # Prefix-only samples prevent future leakage despite noncausal attention.
            corr = corr.masked_fill(~mask[:,None,:,None].bool(), -1e9)
        hidden = self.dropout(corr.softmax(-1)) @ v
        return self.w_layer(hidden.transpose(1,2).contiguous().view(batch,-1,self.num_heads*self.size_head))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, heads, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(heads,hidden_size,dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size,dropout)
        self.input_sublayer = SublayerConnection(hidden_size,dropout)
        self.output_sublayer = SublayerConnection(hidden_size,dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden,lambda x:self.attention(x,x,x,mask))
        return self.dropout(self.output_sublayer(hidden,self.feed_forward))


class Transformer_rep(nn.Module):
    def __init__(self, width, settings):
        super().__init__()
        self.capture = settings.num_blocks-settings.last
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(width,settings.heads,settings.dropout) for _ in range(settings.num_blocks)])

    def forward(self, hidden, mask):
        encoded = None
        for index,block in enumerate(self.transformer_blocks,1):
            hidden = block(hidden,mask)
            if index == self.capture: encoded = hidden
        return hidden,encoded


class EndpointNetwork(nn.Module):
    def __init__(self, width, settings, item_count):
        super().__init__()
        self.hidden_size = width
        self.settings = settings
        self.time_embed = nn.Sequential(nn.Linear(width,width*4),nn.SiLU(),nn.Linear(width*4,width))
        self.att = Transformer_rep(width,settings)
        self.dropout = nn.Dropout(settings.dropout)
        self.norm_fm_rep = LayerNorm(width)
        self.decoder = None
        if settings.mse_weight > 0:
            self.decoder = nn.Sequential(nn.Linear(width,512),nn.Tanh(),nn.Linear(512,2048),
                                         nn.Tanh(),nn.Linear(2048,item_count+1))
        if settings.method == 'history_ce':
            # Common Transformer initial weights and initialization RNG still match.
            del self.time_embed

    def timestep_embedding(self,timesteps):
        half = self.hidden_size//2
        freqs = torch.exp(-math.log(10000)*torch.arange(half,dtype=torch.float32,device=timesteps.device)/half)
        values = timesteps[:,None].float()*freqs[None]
        result = torch.cat([values.cos(),values.sin()],-1)
        return F.pad(result,(0,self.hidden_size%2))

    def forward(self, rep_item, x_t, t, mask_seq, *, fusion_noise=None, decode=False):
        hidden = rep_item
        if self.settings.method != 'history_ce':
            emb_t = self.time_embed(self.timestep_embedding(t))
            if fusion_noise is None:
                scale = self.settings.lambda_uncertainty
                # CPU draws reproduce the reference RNG on CPU; GPU draws avoid
                # a per-ODE-step host allocation and preserve the same distribution.
                fusion_noise = torch.normal(mean=torch.full_like(rep_item,scale),std=torch.full_like(rep_item,scale))
            hidden = rep_item + fusion_noise*(x_t+emb_t).unsqueeze(1)
        hidden,encoded = self.att(hidden,mask_seq)
        endpoint = self.norm_fm_rep(self.dropout(hidden))[:,-1]
        reconstruction = self.decoder(encoded[:,-1]) if decode and self.decoder is not None else None
        return endpoint,reconstruction


def sample_time(settings,batch,device):
    if settings.sampling_method == 'logit_normal':
        t = torch.normal(settings.m_logNorm,settings.s_logNorm,size=(batch,),device=device).sigmoid()
    else:
        u = torch.rand(batch,device=device)
        if settings.sampling_method == 'mode':
            t = 1-u-settings.s_modsamp*((math.pi/2*u).cos().square()-1+u)
        elif settings.sampling_method == 'cosmap':
            t = 1-1/((math.pi/2*u).tan()+1)
        else:t = u
    return t*(1-settings.eps)+settings.eps


def euler_sample(network,history,mask,z0,settings,*,steps=None):
    steps = settings.sample_N if steps is None else steps
    x = z0
    for index in range(steps):
        t = settings.eps_reverse+index/steps*(1-settings.eps_reverse)
        times = x.new_full((len(x),),t*(1/settings.eps_reverse-1))
        endpoint,_ = network(history,x,times,mask)
        # Reference uses endpoint - ORIGINAL z0, not endpoint - current x.
        x = x + (endpoint-z0)/steps
    return x
