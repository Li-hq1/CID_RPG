"""One backward pass jointly updates the online AE and FMRec, using Adam/StepLR."""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from compression.geometry_ae.model import loss_terms
from .training import gradient_norm


class FMRecTrainer:
    def __init__(self,model,train_ids,loader_generator=None):
        self.model = model
        self.config = model.config
        self.train_ids = train_ids.long().cpu()
        self.loader_generator = loader_generator or torch.Generator().manual_seed(self.config.training.seed)
        self.ae_generator = torch.Generator().manual_seed(self.config.training.seed+1)
        t,a = self.config.training,self.config.ae
        self.optimizer = Adam([
            {'params':list(model.rec_parameters()),'lr':t.lr,'weight_decay':t.weight_decay},
            {'params':model.ae.parameters(),'lr':a.lr,'weight_decay':a.weight_decay}],lr=t.lr)
        self.scheduler = StepLR(self.optimizer,step_size=t.decay_step,gamma=t.gamma)
        self.steps = 0
        self.gradient_audit = None

    def step(self,ids,targets):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        objective,terms = self.model.objectives(ids,targets)
        a,t = self.config.ae,self.config.training
        local = objective.new_zeros(())
        if t.local_weight>0:
            indices = torch.randperm(len(self.train_ids),generator=self.ae_generator)[:a.batch_size]
            raw = self.model.raw_items(self.train_ids[indices].to(self.model.device))
            reconstructed,z = self.model.ae(raw)
            geometry = loss_terms(raw,reconstructed,z,self.model.ae.input_scale,a.relation_temperature,
                                  batch_relation=a.relation_mode=='batch')
            weights = {'reconstruction':1.,'cosine':a.cosine_weight,'variance':a.variance_weight,
                       'correlation':a.correlation_weight,'relation':a.relation_weight if a.relation_mode=='batch' else 0.}
            local = sum(geometry[name]*weight for name,weight in weights.items())
        if self.gradient_audit is None:
            self.gradient_audit = {name+'_ae_encoder_grad_norm':gradient_norm(loss,self.model.ae.encoder.parameters())
                                   for name,loss in terms.items()}
            self.gradient_audit['recommendation_ae_encoder_grad_norm'] = gradient_norm(objective,self.model.ae.encoder.parameters())
            if not self.gradient_audit['recommendation_ae_encoder_grad_norm']>0:
                raise RuntimeError('Recommendation gradients did not reach AE encoder.')
        total = objective+t.local_weight*local
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite joint FMRec objective.')
        total.backward()
        for name,parameter in self.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite gradient: '+name)
        self.optimizer.step()
        self.model.invalidate_catalog()
        self.steps += 1
        return {'loss':float(total.detach()),'ae':float(local.detach()),
                **{name:float(value.detach()) for name,value in terms.items()}}

    def state_dict(self):
        return {'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),
                'steps':self.steps,'gradient_audit':self.gradient_audit,
                'loader_rng':self.loader_generator.get_state(),'ae_rng':self.ae_generator.get_state()}

    def load_state_dict(self,state):
        self.optimizer.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler'])
        self.steps = state['steps'];self.gradient_audit = state['gradient_audit']
        self.loader_generator.set_state(state['loader_rng'].cpu());self.ae_generator.set_state(state['ae_rng'].cpu())
