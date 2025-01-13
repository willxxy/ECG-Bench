from tqdm import tqdm
import torch
import wandb
import torch.distributed as dist
import gc
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TrainerConfig:
    """Configuration class for Trainer parameters"""
    model_name: str
    distributed: bool = False
    enable_logging: bool = False
    dev_mode: bool = False
    toy_mode: bool = False
    checkpoint_frequency: int = 50
    max_grad_norm: float = 1.0
    dev_steps: int = 10


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        config: TrainerConfig,
        save_dir: str
    ):
        """
        Initialize the trainer with model, optimizer and configuration.
        
        Args:
            model: The PyTorch model to train
            optimizer: Optimizer with learning rate scheduling
            config: TrainerConfig object containing training parameters
            save_dir: Directory path to save model checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.save_dir = save_dir
        
        # Initialize tracking variables
        self.total_loss = 0
        self.batch_count = 0
        
    def _log_metrics(self, loss: float, epoch: int, step: int) -> None:
        """Log metrics to wandb if logging is enabled"""
        if self.config.enable_logging:
            wandb.log({
                "train_step_loss": loss,
                "epoch": epoch,
                "train_step": step
            })
    
    def _save_checkpoint(self, epoch: int, step: int) -> None:
        """Save model checkpoint with distributed training support"""
        if (step + 1) % self.config.checkpoint_frequency == 0 and not self.config.toy_mode:
            train_model_state_dict = (
                self.model.module.state_dict()
                if self.config.distributed
                else self.model.state_dict()
            )
            
            checkpoint = {
                'model': train_model_state_dict,
                'epoch': epoch
            }
            
            if self.config.distributed:
                dist.barrier()
                if dist.get_rank() == 0:
                    self._save_distributed_checkpoint(checkpoint, epoch, step)
            else:
                self._save_local_checkpoint(checkpoint, epoch, step)
    
    def _save_distributed_checkpoint(self, checkpoint: Dict, epoch: int, step: int) -> None:
        """Save checkpoint in distributed training mode"""
        torch.cuda.empty_cache()
        gc.collect()
        self._save_local_checkpoint(checkpoint, epoch, step)
    
    def _save_local_checkpoint(self, checkpoint: Dict, epoch: int, step: int) -> None:
        """Save checkpoint locally"""
        save_path = f'{self.save_dir}/best_train_model_{epoch}_{step}.pth'
        torch.save(checkpoint, save_path)
        print(f"Best model saved at epoch: {epoch+1} {step}")
    
    def _train_step(self, batch: Any) -> Optional[float]:
        """Execute a single training step"""
        if batch is None:
            return None
            
        try:
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = out.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )
            
            self.optimizer.step_and_update_lr()
            return loss.item()
            
        except Exception as e:
            print(f"Error during training step: {e}")
            return None
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader containing training data
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        if self.config.distributed:
            dataloader.sampler.set_epoch(epoch)
        
        self.total_loss = 0
        self.batch_count = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f'Training {self.config.model_name}',
            position=0,
            leave=True
        )
        
        for step, batch in enumerate(progress_bar):
            loss = self._train_step(batch)
            
            if loss is not None:
                self.total_loss += loss
                self.batch_count += 1
                self._log_metrics(loss, epoch, step)
                self._save_checkpoint(epoch, step)
            
            if self.config.dev_mode and step + 1 >= self.config.dev_steps:
                break
        
        if self.batch_count == 0:
            print("No valid batches for training.")
            average_loss = float('inf')
        else:
            average_loss = self.total_loss / self.batch_count
        
        return {'average_loss': average_loss}