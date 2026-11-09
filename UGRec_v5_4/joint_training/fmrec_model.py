"""Online AE replaces FMRec's item-ID table on every differentiable route."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from compression.geometry_ae.model import GeometryAE
from models.fmrec import LayerNorm, EndpointNetwork, sample_time, euler_sample
from .normalization import LatentNormalizer


class FMRecJointModel(nn.Module):
    def __init__(self,config,raw,ae=None):
        super().__init__()
        config.validate()
        self.config = config
        self.raw_features = torch.as_tensor(raw,dtype=torch.float32,device='cpu').detach().contiguous()
        if self.raw_features.ndim != 2 or not torch.isfinite(self.raw_features).all():
            raise ValueError('Expected finite raw content features.')
        self.n_items = len(raw)+1
        self.ae = ae if ae is not None else GeometryAE(self.raw_features.shape[1],config.ae)
        self.normalizer = LatentNormalizer(config.ae.latent_dim) if config.fmrec.latent_normalization=='layer' else nn.Identity()
        self.embed_dropout = nn.Dropout(config.fmrec.emb_dropout)
        self.LayerNorm = LayerNorm(config.ae.latent_dim)
        self.endpoint = EndpointNetwork(config.ae.latent_dim,config.fmrec,len(raw))
        self.register_buffer('_raw_cache',None,persistent=False)
        self.register_buffer('catalog',None,persistent=False)

    @property
    def device(self):return next(self.parameters()).device

    def rec_parameters(self):
        return (parameter for name,parameter in self.named_parameters() if not name.startswith('ae.'))

    def cache_features(self):self._raw_cache = self.raw_features.to(self.device)

    def raw_items(self,ids):
        return self._raw_cache[ids-1] if self._raw_cache is not None else self.raw_features[ids.cpu()-1].to(self.device)

    def encode_ids(self,ids):
        ids = ids.to(self.device)
        if (ids<0).any() or (ids>=self.n_items).any():raise ValueError('Invalid item ID.')
        unique,inverse = ids.flatten().unique(sorted=True,return_inverse=True)
        positive = unique>0
        z = self.normalizer(self.ae.latent(self.raw_items(unique[positive])))
        lookup = z.new_zeros((len(unique),self.config.ae.latent_dim))
        lookup = lookup.index_copy(0,positive.nonzero(as_tuple=True)[0],z)
        return lookup[inverse].reshape(*ids.shape,self.config.ae.latent_dim)

    def history(self,ids):
        if ids.ndim!=2 or ids.shape[1]!=self.config.data.max_item_seq_len or not (ids[:,-1]>0).all():
            raise ValueError('FMRec requires left-padded nonempty prefix histories.')
        if ((ids[:,:-1]>0)&(ids[:,1:]==0)).any():raise ValueError('Padding must be on the left.')
        return self.LayerNorm(self.embed_dropout(self.encode_ids(ids))),ids>0

    def objectives(self,ids,targets):
        settings = self.config.fmrec
        history,mask = self.history(ids)
        # Live catalog graph: CE gradients reach the AE for every item.
        catalog = self.encode_ids(torch.arange(1,self.n_items,device=self.device))
        clean = catalog[targets-1]  # FM target and bridge are ALSO live, no detach.
        if settings.method=='history_ce':
            prediction,reconstruction = self.endpoint(history,None,None,mask)
        else:
            t = sample_time(settings,len(targets),self.device)
            z0 = torch.randn_like(clean)
            xt = t[:,None]*clean+(1-t[:,None])*z0
            prediction,reconstruction = self.endpoint(history,xt,t*(1/settings.eps-1),mask,
                                                       decode=settings.mse_weight>0)
        ce = checkpoint(lambda q,k:F.cross_entropy(q@k.T,targets-1),prediction,catalog,use_reentrant=False)
        terms = {'ce':ce}
        objective = settings.ce_weight*ce
        if settings.method!='history_ce':
            terms['fm'] = F.mse_loss(prediction,clean)
            objective = objective+settings.fm_weight*terms['fm']
        if settings.mse_weight>0:
            terms['history_mse'] = balanced_history_mse(ids,reconstruction,settings.mask_ratio)
            objective = objective+settings.mse_weight*terms['history_mse']
        return objective,terms

    def invalidate_catalog(self):self.catalog=None

    @torch.no_grad()
    def refresh_catalog(self):
        self.catalog = self.encode_ids(torch.arange(1,self.n_items,device=self.device))

    @torch.no_grad()
    def predict_query(self,ids,*,steps=None,z0=None):
        if self.training:raise RuntimeError('Call eval before inference.')
        history,mask = self.history(ids)
        if self.config.fmrec.method=='history_ce':
            return self.endpoint(history,None,None,mask)[0]
        if z0 is None:z0 = torch.randn(len(ids),self.config.ae.latent_dim,device=self.device)
        return euler_sample(self.endpoint,history,mask,z0,self.config.fmrec,steps=steps)

    @torch.no_grad()
    def log_probabilities(self,ids,*,steps=None):
        if self.catalog is None:raise RuntimeError('Refresh evaluation catalog first.')
        return (self.predict_query(ids,steps=steps)@self.catalog.T).log_softmax(-1)


def balanced_history_mse(ids,output,mask_ratio):
    # Matches the reference: all positive user-item entries and uniformly sampled
    # zero entries WITH replacement; mean is over the full [B,N+1] tensor.
    target = torch.zeros_like(output)
    rows = torch.arange(len(ids),device=ids.device)[:,None].expand_as(ids)
    positive = ids>0
    target[rows[positive],ids[positive]]=1
    zeros = (target==0).nonzero(as_tuple=False)
    count = min(len(zeros),int(target.sum().item()*mask_ratio))
    mask = target.bool()
    if count:
        selected = zeros[torch.randint(len(zeros),(count,),device=ids.device)]
        mask[selected[:,0],selected[:,1]]=True
    return F.mse_loss(output*mask,target*mask)
