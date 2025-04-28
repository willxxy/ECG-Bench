import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model
from collections import namedtuple

class DINOv2(nn.Module):
    def __init__(self, dinov2):
        super(DINOv2, self).__init__()
        self.dinov2 = dinov2
        self.momentum_teacher = None
        self.momentum_scheduler = None
        self.center = None
        self.teacher_temp = 0.04
        self.student_temp = 0.1
        self.center_momentum = 0.9
        self.embedding_dim = 768
            
    def forward(self, batch):
        if self.training:
            student_output = self.dinov2(pixel_values=batch['dinov2_pixel'].to(self.dinov2.device))
            student_embeddings = student_output.last_hidden_state.mean(dim=1)
            
            with torch.no_grad():
                if self.momentum_teacher is None:
                    self.momentum_teacher = self._init_teacher()
                self._update_teacher()
                
                teacher_output = self.momentum_teacher(pixel_values=batch['dinov2_pixel'].to(self.dinov2.device))
                teacher_embeddings = teacher_output.last_hidden_state.mean(dim=1)
                
                if self.center is None:
                    self.center = torch.zeros_like(teacher_embeddings[0])
                self._update_center(teacher_embeddings)
                
                teacher_embeddings = F.normalize(teacher_embeddings - self.center, dim=-1)
            
            student_embeddings = F.normalize(student_embeddings, dim=-1)
            loss = self._compute_loss(student_embeddings, teacher_embeddings)
            
            Output = namedtuple('Output', ['loss', 'embeddings'])
            return Output(loss=loss, embeddings=student_embeddings)
        else:
            out = self.dinov2(pixel_values=batch['dinov2_pixel'].to(self.dinov2.device))
            return out
    
    def _init_teacher(self):
        teacher = Dinov2Model.from_pretrained("facebook/dinov2-base")
        teacher.load_state_dict(self.dinov2.state_dict())
        teacher.eval()
        return teacher
    
    def _update_teacher(self):
        m = self.momentum_scheduler if self.momentum_scheduler is not None else 0.996
        for param_q, param_k in zip(self.dinov2.parameters(), self.momentum_teacher.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
    
    def _update_center(self, teacher_embeddings):
        batch_center = torch.mean(teacher_embeddings, dim=0)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def _compute_loss(self, student_embeddings, teacher_embeddings):
        """Compute DINOv2 loss"""
        student_out = student_embeddings / self.student_temp
        teacher_out = teacher_embeddings / self.teacher_temp
        
        loss = -torch.sum(F.softmax(teacher_out, dim=-1) * F.log_softmax(student_out, dim=-1), dim=-1)
        return loss.mean()
    
    @torch.no_grad()
    def get_embeddings(self, batch):
        """Get embeddings for inference"""
        self.eval()
        out = self.dinov2(pixel_values=batch['dinov2_pixel'].to(self.dinov2.device))
        return out.pooler_output
        
        