import numpy as np

class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
    
    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        step = max(1, self.n_current_steps)
        
        d_step = 1.0 / np.sqrt(step)
        
        if self.n_warmup_steps > 0:
            d_warmup = step * np.power(self.n_warmup_steps, -1.5)
        else:
            return d_step
        
        return min(d_step, d_warmup)

    def _update_learning_rate(self):
        lr = self.init_lr * self._get_lr_scale()
        
        min_lr = 1e-8
        lr = max(lr, min_lr)
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        
        self.n_current_steps += 1
            
    @property
    def learning_rate(self):
        "Get current learning rate"
        return self._optimizer.param_groups[0]['lr']