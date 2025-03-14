from tqdm import tqdm
import torch
import torch.distributed as dist
import gc
import wandb

def post_trainer_dpo(model, dataloader, tokenizer, args, optimizer, epoch, blender, dpo, ref_model):
    if args.dis:
        dataloader.sampler.set_epoch(epoch)
    total_loss = 0.0
    len_of_batch = 0
    dev_count = 0
    gt_answers1 = []
    gen_answers1 = []
    gt_answers2 = []
    gen_answers2 = []
    
    progress_bar = tqdm(dataloader, desc=f'Post-Training {args.model}')
    
    chat_histories = []  # To store chat histories for each conversation
    
    for step, batch in enumerate(progress_bar):
        if batch is None:
            print(f"Skipping invalid batch")
            continue
        
        try:
            gt_input_ids1 = batch['input_ids']
            gt_attention_mask1 = batch['attn_mask']
            
            gt_input_ids2 = batch['input_ids']
            gt_attention_mask2 = batch['attn_mask']
            
            chat_input_ids1 = gt_input_ids1.clone()
            chat_input_ids2 = gt_input_ids2.clone()
            
            chat_attention_mask1 = gt_attention_mask1.clone()
            chat_attention_mask2 = gt_attention_mask2.clone()
            
            assistant_ranges = batch['assistant_ranges']
            
            if args.inference == 'second':
                encoder_out = batch['encoder_out']
                signal_id_index = batch['signal_id_index'].item()
                
            offset1 = 0
            offset2 = 0
            
            # Initialize chat history for this batch
            current_chat_history = []
            
            for conv_turn in assistant_ranges:
                start1 = conv_turn['start'] + 4 + offset1
                end1 = conv_turn['end'] + 1 + offset1
                
                start2 = conv_turn['start'] + 4 + offset2
                end2 = conv_turn['end'] + 1 + offset2
                
                curr_input_ids1 = chat_input_ids1[:, :start1]
                curr_attention_mask1 = chat_attention_mask1[:, :start1]
                
                curr_input_ids2 = chat_input_ids2[:, :start2]
                curr_attention_mask2 = chat_attention_mask2[:, :start2]
            
                # Get the prompt for this turn (user's question)
                prompt = tokenizer.decode(curr_input_ids1[0, :], skip_special_tokens=False)
                current_chat_history.append(prompt)
                
                # Set to eval mode for generation
                model.eval()
                with torch.no_grad():
                    if args.inference == 'second':
                        out1 = model.generate_chat(
                            input_ids=curr_input_ids1,
                            attention_mask=curr_attention_mask1,
                            tokenizer=tokenizer,
                            encoder_out=encoder_out,
                            signal_id_index=signal_id_index)
                        
                        out2 = model.generate_chat(
                            input_ids=curr_input_ids2,
                            attention_mask=curr_attention_mask2,
                            tokenizer=tokenizer,
                            encoder_out=encoder_out,
                            signal_id_index=signal_id_index)
                    else:
                        out1 = model.generate_chat(
                            input_ids=curr_input_ids1,
                            attention_mask=curr_attention_mask1,
                            tokenizer=tokenizer)
                        
                        out2 = model.generate_chat(
                            input_ids=curr_input_ids2,
                            attention_mask=curr_attention_mask2,
                            tokenizer=tokenizer)
                
                # Set back to train mode for loss computation
                model.train()
                
                chat_input_ids1 = torch.cat([
                    chat_input_ids1[:, :start1],
                    out1[:, start1:].cpu(),
                    gt_input_ids1[:, end1-offset1:]
                ], dim=1)
                
                chat_input_ids2 = torch.cat([
                    chat_input_ids2[:, :start2],
                    out2[:, start2:].cpu(),
                    gt_input_ids2[:, end2-offset2:]
                ], dim=1)
                
                # print('chat_input_ids', tokenizer.decode(chat_input_ids[0]))
                chat_attention_mask1 = torch.ones_like(chat_input_ids1)
                chat_attention_mask2 = torch.ones_like(chat_input_ids2)
                decoded_out1 = tokenizer.batch_decode(out1[:, start1:], skip_special_tokens=False)[0]
                gt_out1 = tokenizer.batch_decode(gt_input_ids1[:, start1-offset1:end1-offset1], skip_special_tokens=False)[0]
                gt_answers1.append(gt_out1)
                gen_answers1.append(decoded_out1)
                
                decoded_out2 = tokenizer.batch_decode(out2[:, start2:], skip_special_tokens=False)[0]
                gt_out2 = tokenizer.batch_decode(gt_input_ids2[:, start2-offset2:end2-offset2], skip_special_tokens=False)[0]
                gt_answers2.append(gt_out2)
                gen_answers2.append(decoded_out2)
                
                offset1 += out1[:, start1:].size(1) - (end1 - start1)
                offset2 += out2[:, start2:].size(1) - (end2 - start2)
                
                # Compare the two generated responses using the current prompt
                judge_results = blender.compare([prompt], 
                                               [decoded_out1],
                                               [decoded_out2])
                
                if judge_results[0] == True:
                    # Use the full sequence with generated output for model 1
                    preferred_output_ids = chat_input_ids1[:, :start1+out1[:, start1:].size(1)].to(model.device)
                    preferred_output_attention_mask = chat_attention_mask1[:, :start1+out1[:, start1:].size(1)].to(model.device)
                    # Use the full sequence with generated output for model 2
                    dispreferred_output_ids = chat_input_ids2[:, :start2+out2[:, start2:].size(1)].to(model.device)
                    dispreferred_output_attention_mask = chat_attention_mask2[:, :start2+out2[:, start2:].size(1)].to(model.device)
                else:
                    # Use the full sequence with generated output for model 2
                    preferred_output_ids = chat_input_ids2[:, :start2+out2[:, start2:].size(1)].to(model.device)
                    preferred_output_attention_mask = chat_attention_mask2[:, :start2+out2[:, start2:].size(1)].to(model.device)
                    # Use the full sequence with generated output for model 1
                    dispreferred_output_ids = chat_input_ids1[:, :start1+out1[:, start1:].size(1)].to(model.device)
                    dispreferred_output_attention_mask = chat_attention_mask1[:, :start1+out1[:, start1:].size(1)].to(model.device)
                
                preferred_log_prob = dpo.get_log_prob(model.llm(input_ids = preferred_output_ids, attention_mask = preferred_output_attention_mask).logits, preferred_output_ids)
                dispreferred_log_prob = dpo.get_log_prob(model.llm(input_ids = dispreferred_output_ids, attention_mask = dispreferred_output_attention_mask).logits, dispreferred_output_ids)
                
                with torch.no_grad():
                    ref_preferred_log_prob = dpo.get_log_prob(ref_model.llm(input_ids = preferred_output_ids, attention_mask = preferred_output_attention_mask).logits, preferred_output_ids)
                    ref_dispreferred_log_prob = dpo.get_log_prob(ref_model.llm(input_ids = dispreferred_output_ids, attention_mask = dispreferred_output_attention_mask).logits, dispreferred_output_ids)
                    
                loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = dpo.calculate_DPO_loss(preferred_log_prob.to(model.device), dispreferred_log_prob.to(model.device),
                            ref_preferred_log_prob.to(model.device), ref_dispreferred_log_prob.to(model.device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step_and_update_lr()
                    
                total_loss += loss.item()
                len_of_batch += 1
                print("Average Loss:",total_loss / len_of_batch, end = "\r")
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
            
            # Store chat history for this batch
            chat_histories.append(current_chat_history)

        except Exception as e:
            print('\nError occurred during evaluation:')
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Error location: {e.__traceback__.tb_lineno}")


    if len_of_batch == 0:
        print("No valid batches for training.")
        average_loss = float('inf')
    else:
        average_loss = total_loss / len_of_batch

    return {'average_loss': average_loss}