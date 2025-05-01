from tqdm import tqdm
import torch
import wandb
import torch.distributed as dist
import gc

def trainer(model, dataloader, optimizer, args, epoch):
    model.train()
    if args.dis:
        dataloader.sampler.set_epoch(epoch)
        show_progress = dist.get_rank() == 0
    else:
        show_progress = True

    total_loss = 0.0
    len_of_batch = 0
    dev_count = 0
    
    progress_bar = tqdm(dataloader, desc=f'Training {args.model}', disable=not show_progress)
    
    for step, batch in enumerate(progress_bar):
        if batch is None:
            if show_progress:
                print(f"Skipping invalid batch at step {step}")
            continue
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()

        if args.model in ['clip', 'vit', 'merl', 'stmem', 'siglip', 'mtae', 'mlae']:
            pass
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        len_of_batch += 1

        if args.log:
            wandb.log({"train_step_loss": loss.item(), "epoch": epoch, "train_step": step})

        if (step + 1) % 50000 == 0:
            if args.dis:
                dist.barrier()
                if dist.get_rank() == 0:
                    model_state_dict = model.module.state_dict()
                    train_checkpoint = {
                        'model': model_state_dict,
                        'epoch': epoch
                    }
                    torch.cuda.empty_cache()
                    gc.collect()
                    checkpoint_path = f"{args.save_path}/model_{epoch}_{step}.pth"
                    torch.save(train_checkpoint, checkpoint_path)
                    print(f"Model saved at epoch: {epoch+1}, step: {step}")
                dist.barrier()
            else:
                model_state_dict = model.state_dict()
                train_checkpoint = {
                    'model': model_state_dict,
                    'epoch': epoch
                }
                checkpoint_path = f"{args.save_path}/model_{epoch}_{step}.pth"
                torch.save(train_checkpoint, checkpoint_path)
                print(f"Model saved at epoch: {epoch+1}, step: {step}")

        if args.dev:
            dev_count += 1
            if dev_count == 10:
                break

    average_loss = total_loss / len_of_batch if len_of_batch > 0 else float('inf')
    return {'average_loss': average_loss}
