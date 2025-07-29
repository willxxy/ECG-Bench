import numpy as np
import torch.distributed as dist

def _effective_global_bs(args):
    if dist.is_initialized():
        ws = dist.get_world_size()
    elif args.dis:
        ws = len(args.gpus.split(','))
    else:
        ws = 1
    ga = getattr(args, 'grad_accum_steps', 1)
    return int(args.batch_size) * int(ws) * int(ga)

class ScheduledOptim():
    def __init__(self, optimizer, d_model, args):
        self._optimizer = optimizer
        self.n_warmup_steps = args.warmup
        self.n_current_steps = 0
        self.args = args

        eff_bs = _effective_global_bs(args)

        ref_bs = getattr(args, 'ref_global_bs', None)
        if ref_bs is None:
            ref_bs = int(getattr(args, 'batch_size', 1)) * int(getattr(args, 'grad_accum_steps', 1))
        ref_bs = max(int(ref_bs), 1)

        scale = max(eff_bs / ref_bs, 1e-8)

        if args.train == 'first':
            peak_lr = args.lr * scale
            self.init_lr = peak_lr * (self.n_warmup_steps ** 0.5)
        else:
            self.init_lr = (d_model ** -0.5) * scale

        wd_mode = getattr(args, 'scale_wd', 'none')
        wd_scale = 1.0 if wd_mode=='none' else (1.0/(scale**0.5) if wd_mode=='inv_sqrt' else 1.0/scale)
        for g in self._optimizer.param_groups:
            if 'weight_decay' in g and g['weight_decay'] is not None:
                g['weight_decay'] = float(g['weight_decay']) * wd_scale

        if (not getattr(args,'dis',False)) or (dist.is_initialized() and dist.get_rank()==0):
            print(f"[scale] eff_bs={eff_bs}, ref_bs={ref_bs}, scale={scale:.4g}, wd_mode={wd_mode}, init_lr={self.init_lr:.3e}")

    def step_and_update_lr(self):
        self._update_learning_rate(); self._optimizer.step()
        
    def zero_grad(self): self._optimizer.zero_grad()
    
    def _get_lr_scale(self):
        step = max(1, self.n_current_steps)
        d_step = 1.0 / np.sqrt(step)
        return min(d_step, step * (self.n_warmup_steps ** -1.5)) if self.n_warmup_steps>0 else d_step
    
    def _update_learning_rate(self):
        lr = max(self.init_lr * self._get_lr_scale(), 1e-8)
        for g in self._optimizer.param_groups: g['lr'] = lr
        self.n_current_steps += 1
    
    @property
    def learning_rate(self): return self._optimizer.param_groups[0]['lr']
