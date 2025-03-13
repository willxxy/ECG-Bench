import torch
import torch.nn.functional as F

class DPO:
    def __init__(self, beta=0.5):
        self.beta = beta
    
    def calculate_DPO_loss(self, model_prefered_logprob, model_disprefered_logprob,
                          ref_prefered_logprob, ref_disprefered_logprob):
        
        prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
        disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

        reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
        reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

        loss = -F.logsigmoid(self.beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

        return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

    @staticmethod
    def get_log_prob(logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)