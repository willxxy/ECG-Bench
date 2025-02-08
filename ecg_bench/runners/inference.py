import torch
from tqdm import tqdm
import numpy as np

def tester(model, dataloader, tokenizer, args, train_utils):
    model.eval()
    len_of_batch = 0
    dev_count = 0
    gt_answers = []
    gen_answers = []
    questions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = f'Testing {args.model}', position=0, leave=True):
            if batch is None:
                print(f"Skipping invalid batch ")
                continue
            
            answer = batch['answer']
            
            try:
                out = [model.generate(batch, tokenizer)]
                gt_answers.append(answer[0])
                gen_answers.append(out[0])
                questions.append(batch['question'][0])
            except Exception as e:
                print('could not evaluate for some reason:', str(e))
                print(f"Error type: {type(e).__name__}")
                print(out)
                print(answer)
                gt_answers.append("")
                gen_answers.append("")
                questions.append(batch['question'][0])
            
            len_of_batch += 1
            
            if args.dev:
                dev_count += 1
                if dev_count == 25:
                    break

    try:
        all_metrics = train_utils.evaluate_strings(gt_answers, gen_answers, args.device)
    except Exception as e:
        print('Error during batch evaluation:', str(e))
        all_metrics = {
            'BLEU': 0, 
            'METEOR': 0.0, 
            'ROUGE': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}, 
            'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]}
        }
    
    return {
        'metrics': all_metrics,
        'qa_results': {
            'questions': questions,
            'gt_answers': gt_answers,
            'gen_answers': gen_answers
        }
    }
    
    
    
# def tester_chat(model, dataloader, tokenizer, args, train_utils):
#     model.eval()
#     len_of_batch = 0
#     dev_count = 0
#     gt_answers = []
#     gen_answers = []
#     questions = []
    
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Testing {args.model}', position=0, leave=True)):
#             if batch is None:
#                 print(f"Skipping invalid batch")
#                 continue
            
#             try:
#                 # For each conversation in the batch
#                 for b_idx in range(len(batch['input_ids'])):
#                     print(f"\nConversation {b_idx} in batch:")
#                     print(f"{'_'*30}")
                    
#                     input_ids = batch['input_ids'][b_idx].unsqueeze(0)
#                     attention_mask = batch['attn_mask'][b_idx].unsqueeze(0)
                    
#                     # Keep track of the conversation history for this example
#                     current_input_ids = input_ids.clone()
#                     current_attention_mask = attention_mask.clone()
                    
#                     # Get all assistant ranges for this conversation
#                     assistant_ranges = batch['assistant_ranges']  # This is now a list of dicts
                    
#                     print(f"Number of turns: {len(assistant_ranges)}")
#                     print('Assistant ranges:', assistant_ranges)
                    
#                     # Process each turn in the conversation
#                     for turn_idx, range_dict in enumerate(assistant_ranges):
#                         start_idx = range_dict['start'].item()
#                         end_idx = range_dict['end'].item()
#                         eot_idx = range_dict['eot'].item()
                        
#                         print(f"\nProcessing turn {turn_idx + 1}/{len(assistant_ranges)}")
#                         print(f"Indices - Start: {start_idx}, End: {end_idx}, EOT: {eot_idx}")
                        
#                         # Get context up to this turn
#                         context = tokenizer.decode(current_input_ids[0][:start_idx], skip_special_tokens=True)
#                         print(f"\nContext for this turn:")
#                         print(f"{context}...")
                        
#                         # Generate response for this turn
#                         curr_input_ids = current_input_ids[:, :start_idx]
#                         curr_attention_mask = current_attention_mask[:, :start_idx]
                        
                        
#                         print('curr_input_ids.shape', curr_input_ids.shape)
#                         print('curr_attention_mask.shape', curr_attention_mask.shape)
#                         print("\nGenerating response...")
#                         outputs = model.generate_chat(
#                             input_ids=curr_input_ids,
#                             attention_mask=curr_attention_mask,
#                             tokenizer=tokenizer
#                         )
                        
#                         # Get the generated response and ground truth
#                         generated_response = tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=False)
#                         ground_truth = tokenizer.decode(current_input_ids[0][start_idx:eot_idx+1], skip_special_tokens=False)
                        
#                         print("\nGround Truth:")
#                         print(f"{ground_truth}...")
#                         print("\nGenerated Response:")
#                         print(f"{generated_response}...")
                        
#                         # Store results
#                         gt_answers.append(ground_truth)
#                         gen_answers.append(generated_response)
#                         questions.append(context)
                        
#                         # Update conversation history with generated response for next turn
#                         if turn_idx < len(assistant_ranges) - 1:
#                             next_start = assistant_ranges[turn_idx + 1]['start'].item()
#                             current_input_ids = torch.cat([
#                                 current_input_ids[:, :start_idx].cpu(),
#                                 outputs[0][start_idx:].unsqueeze(0).cpu(),
#                                 current_input_ids[:, eot_idx+1:next_start].cpu()
#                             ], dim=1)
#                             current_attention_mask = torch.ones_like(current_input_ids).cpu()
#                             print("\nConversation structures after turn", turn_idx + 1)
#                             print("-" * 50)
#                             print("Ground Truth conversation:")
#                             print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
#                             print("-" * 50)
#                             print("Generated conversation:")
#                             print(tokenizer.decode(current_input_ids[0], skip_special_tokens=False))
#                             print("-" * 50)
                        
#                         print(f"\nStored results for turn {turn_idx + 1}")
#                     input('END OF ONE CONVERSATION')
#             except Exception as e:
#                 print('\nError occurred during evaluation:')
#                 print(f"Error type: {type(e).__name__}")
#                 print(f"Error message: {str(e)}")
#                 print(f"Error location: {e.__traceback__.tb_lineno}")
#                 gt_answers.append("")
#                 gen_answers.append("")
#                 questions.append("")
            
#             len_of_batch += 1
#             print(f"\nCompleted batch {batch_idx}. Total conversations processed: {len_of_batch}")
            
#             if args.dev:
#                 dev_count += 1
#                 if dev_count == 25:
#                     print("\nDev mode: Stopping after 25 batches")
#                     break

#     print("\nCalculating metrics...")
#     try:
#         all_metrics = train_utils.evaluate_strings(gt_answers, gen_answers, args.device)
#         print("\nMetrics calculated successfully:")
#         print(f"BLEU: {all_metrics['BLEU']}")
#         print(f"METEOR: {all_metrics['METEOR']}")
#         print(f"ROUGE-L: {all_metrics['ROUGE']['rouge-l']}")
#         print(f"BERTScore F1: {np.mean(all_metrics['BERTSCORE']['hf-f1'])}")
#     except Exception as e:
#         print('\nError during metric calculation:')
#         print(f"Error type: {type(e).__name__}")
#         print(f"Error message: {str(e)}")
#         all_metrics = {
#             'BLEU': 0, 
#             'METEOR': 0.0, 
#             'ROUGE': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}, 
#             'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]}
#         }
    
#     print("\nEvaluation complete!")
#     return {
#         'metrics': all_metrics,
#         'qa_results': {
#             'questions': questions,
#             'gt_answers': gt_answers,
#             'gen_answers': gen_answers
#         }
#     }
    
    
    
def tester_chat(model, dataloader, tokenizer, args, train_utils):
    model.eval()
    len_of_batch = 0
    dev_count = 0
    gt_answers = []
    gen_answers = []
    questions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Testing {args.model}', position=0, leave=True)):
            if batch is None:
                print(f"Skipping invalid batch")
                continue
            
            # try:
            # For each conversation in the batch
            gt_input_ids = batch['input_ids']
            gt_attention_mask = batch['attn_mask']
            
            chat_input_ids = gt_input_ids.clone()
            chat_attention_mask = gt_attention_mask.clone()
            
            assistant_ranges = batch['assistant_ranges']
            # print('input_ids', input_ids)
            # print('input_ids.shape', gt_input_ids.shape)
            # print('attention_mask', attention_mask)
            # print('attention_mask.shape', gt_attention_mask.shape)
            # print('assistant_ranges', assistant_ranges)
            # input('eweeeeeee')
            offset = 0
            for conv_turn in assistant_ranges:
                start = conv_turn['start'] + 4 + offset # skip start header, assist, end header, and \n
                end = conv_turn['end'] + 1 + offset
                print('start', start)
                print('end', end)
                # input('eweeeeeee')
                
                curr_input_ids = chat_input_ids[:, :start]
                curr_attention_mask = chat_attention_mask[:, :start]
                
                # print('curr_input_ids.shape', curr_input_ids.shape)
                # print('curr_attention_mask.shape', curr_attention_mask.shape)
                print('curr_input_ids', tokenizer.decode(curr_input_ids[0]))
                
                out = model.generate_chat(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attention_mask,
                    tokenizer=tokenizer
                )
                
                chat_input_ids = torch.cat([
                    chat_input_ids[:, :start],  # Keep conversation history up to start
                    out[:, start:].cpu(),             # Add model's generated response
                    gt_input_ids[:, end-offset:]       # Add remaining conversation prompts
                ], dim=1)
                
                # print('chat_input_ids.shape', chat_input_ids.shape)
                print('chat_input_ids', tokenizer.decode(chat_input_ids[0]))
                # Update attention mask to match new input_ids length
                chat_attention_mask = torch.ones_like(chat_input_ids)
                
                decoded_out = tokenizer.batch_decode(out[:, start:], skip_special_tokens=False)[0]
                gt_out = tokenizer.batch_decode(gt_input_ids[:, start-offset:end-offset], skip_special_tokens=False)[0]
                gt_answers.append(gt_out)
                gen_answers.append(decoded_out)
                offset += out[:, start:].size(1) - (end - start)
                print('decoded_out', decoded_out)
                print('gt_out', gt_out)
                print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            input('-------------------------------------------------------------------------------')
                
                

            # for b_idx in range(len(batch['input_ids'])):
            #     print(f"\nConversation {b_idx} in batch:")
            #     print(f"{'_'*30}")
                
            #     input_ids = batch['input_ids'][b_idx].unsqueeze(0)
            #     attention_mask = batch['attn_mask'][b_idx].unsqueeze(0)
                
            #     # Get all assistant ranges for this conversation
            #     assistant_ranges = batch['assistant_ranges']  # This is now a list of dicts
                
            #     print(f"Number of turns: {len(assistant_ranges)}")
            #     print('Assistant ranges:', assistant_ranges)
                
                # # Process each turn in the conversation
                # for turn_idx, range_dict in enumerate(assistant_ranges):
                #     start_idx = range_dict['start'].item()
                #     end_idx = range_dict['end'].item()
                #     eot_idx = range_dict['eot'].item()
                    
                #     print(f"\nProcessing turn {turn_idx + 1}/{len(assistant_ranges)}")
                #     print(f"Indices - Start: {start_idx}, End: {end_idx}, EOT: {eot_idx}")
                    
                #     # Get context up to this turn
                #     context = tokenizer.decode(current_input_ids[0][:start_idx], skip_special_tokens=True)
                #     print(f"\nContext for this turn:")
                #     print(f"{context}...")
                    
                #     # Generate response for this turn
                #     curr_input_ids = current_input_ids[:, :start_idx]
                #     curr_attention_mask = current_attention_mask[:, :start_idx]
                    
                    
                #     print('curr_input_ids.shape', curr_input_ids.shape)
                #     print('curr_attention_mask.shape', curr_attention_mask.shape)
                #     print("\nGenerating response...")
                #     outputs = model.generate_chat(
                #         input_ids=curr_input_ids,
                #         attention_mask=curr_attention_mask,
                #         tokenizer=tokenizer
                #     )
                    
                #     # Get the generated response and ground truth
                #     generated_response = tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=False)
                #     ground_truth = tokenizer.decode(current_input_ids[0][start_idx:eot_idx+1], skip_special_tokens=False)
                    
                #     print("\nGround Truth:")
                #     print(f"{ground_truth}...")
                #     print("\nGenerated Response:")
                #     print(f"{generated_response}...")
                    
                #     # Store results
                #     gt_answers.append(ground_truth)
                #     gen_answers.append(generated_response)
                #     questions.append(context)
                    
                #     # Update conversation history with generated response for next turn
                #     if turn_idx < len(assistant_ranges) - 1:
                #         next_start = assistant_ranges[turn_idx + 1]['start'].item()
                #         current_input_ids = torch.cat([
                #             current_input_ids[:, :start_idx].cpu(),
                #             outputs[0][start_idx:].unsqueeze(0).cpu(),
                #             current_input_ids[:, eot_idx+1:next_start].cpu()
                #         ], dim=1)
                #         current_attention_mask = torch.ones_like(current_input_ids).cpu()
                #         print("\nConversation structures after turn", turn_idx + 1)
                #         print("-" * 50)
                #         print("Ground Truth conversation:")
                #         print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
                #         print("-" * 50)
                #         print("Generated conversation:")
                #         print(tokenizer.decode(current_input_ids[0], skip_special_tokens=False))
                #         print("-" * 50)
                    
                #     print(f"\nStored results for turn {turn_idx + 1}")
                # input('END OF ONE CONVERSATION')
            # except Exception as e:
            #     print('\nError occurred during evaluation:')
            #     print(f"Error type: {type(e).__name__}")
            #     print(f"Error message: {str(e)}")
            #     print(f"Error location: {e.__traceback__.tb_lineno}")
            #     gt_answers.append("")
            #     gen_answers.append("")
            #     questions.append("")
            
            len_of_batch += 1
            print(f"\nCompleted batch {batch_idx}. Total conversations processed: {len_of_batch}")
            
            if args.dev:
                dev_count += 1
                if dev_count == 25:
                    print("\nDev mode: Stopping after 25 batches")
                    break

    print("\nCalculating metrics...")
    try:
        all_metrics = train_utils.evaluate_strings(gt_answers, gen_answers, args.device)
        print("\nMetrics calculated successfully:")
        print(f"BLEU: {all_metrics['BLEU']}")
        print(f"METEOR: {all_metrics['METEOR']}")
        print(f"ROUGE-L: {all_metrics['ROUGE']['rouge-l']}")
        print(f"BERTScore F1: {np.mean(all_metrics['BERTSCORE']['hf-f1'])}")
    except Exception as e:
        print('\nError during metric calculation:')
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        all_metrics = {
            'BLEU': 0, 
            'METEOR': 0.0, 
            'ROUGE': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}, 
            'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]}
        }
    
    print("\nEvaluation complete!")
    return {
        'metrics': all_metrics,
        'qa_results': {
            'questions': questions,
            'gt_answers': gt_answers,
            'gen_answers': gen_answers
        }
    }