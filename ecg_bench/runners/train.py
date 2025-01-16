from tqdm import tqdm
import torch
import wandb
import torch.distributed as dist
import gc

def trainer(model, dataloader, optimizer, args, epoch):
    model.train()
    if args.dis:
        dataloader.sampler.set_epoch(epoch)

    total_loss = 0.0
    len_of_batch = 0
    dev_count = 0
    
    progress_bar = tqdm(dataloader, desc=f'Training {args.model}')
    
    for step, batch in enumerate(progress_bar):
        if batch is None:
            print(f"Skipping invalid batch at step {step}")
            continue
        
        try:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.loss
            loss.backward()

            if args.model in ['clip', 'vit']:
                pass
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            len_of_batch += 1

            if args.log:
                wandb.log({"train_step_loss": loss.item(), "epoch": epoch, "train_step": step})

            if (step + 1) % 50000 == 0:
                model_state_dict = model.module.state_dict() if args.dis else model.state_dict()
                
                train_checkpoint = {
                    'model': model_state_dict,
                    'epoch': epoch
                }
                
                if args.dis:
                    dist.barrier()
                    if dist.get_rank() == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        checkpoint_path = f"{args.save_path}/model_{epoch}_{step}.pth"
                        torch.save(train_checkpoint, checkpoint_path)
                        print(f"Model saved at epoch: {epoch+1}, step: {step}")
                else:
                    checkpoint_path = f"{args.save_path}/model_{epoch}_{step}.pth"
                    torch.save(train_checkpoint, checkpoint_path)
                    print(f"Model saved at epoch: {epoch+1}, step: {step}")

            if args.dev:
                dev_count += 1
                if dev_count == 10:
                    break

        except Exception as e:
            print(f"Error during training at step {step}: {e}")
            continue

    if len_of_batch == 0:
        print("No valid batches for training.")
        average_loss = float('inf')
    else:
        average_loss = total_loss / len_of_batch

    return {'average_loss': average_loss}
