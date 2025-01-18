import torch.nn as nn

class MERL(nn.Module):
    def __init__(self, merl, args):
        super(MERL, self).__init__()
        self.merl = merl
        self.args = args
    def forward(self, batch):
        if self.args.train == 'first':
            out = self.merl(signal = batch['signal'].to(self.merl.device),
                            input_ids = batch['lm_input_ids'].to(self.merl.device),
                            attention_mask = batch['lm_att_mask'].to(self.merl.device))
        elif self.args.train == 'second':
            out = self.merl(signal = batch['signal'].to(self.merl.device))
        return out