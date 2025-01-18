import torch.nn as nn

class MERL(nn.Module):
    def __init__(self, merl):
        super(MERL, self).__init__()
        self.merl = merl
    def forward(self, batch):
        if self.merl.args.train == 'first':
            out = self.merl(signal = batch['signal'].to(self.merl.device),
                            input_ids = batch['merl_input_ids'].to(self.merl.device),
                            attention_mask = batch['merl_att_mask'].to(self.merl.device))
        elif self.merl.args.train == 'second':
            out = self.merl(signal = batch['signal'].to(self.merl.device))
        return out