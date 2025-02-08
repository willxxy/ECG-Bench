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
            
            try:
                # For each conversation in the batch
                for b_idx in range(len(batch['input_ids'])):
                    print(f"\nConversation {b_idx} in batch:")
                    print(f"{'_'*30}")
                    
                    input_ids = batch['input_ids'][b_idx].unsqueeze(0).cuda()
                    attention_mask = batch['attn_mask'][b_idx].unsqueeze(0).cuda()
                    
                    # Keep track of the conversation history for this example
                    current_input_ids = input_ids.clone()
                    current_attention_mask = attention_mask.clone()
                    
                    # Find all assistant responses in the conversation
                    all_start_indices = batch['assistant_ranges'][b_idx]['start']
                    all_end_indices = batch['assistant_ranges'][b_idx]['end']
                    all_eot_indices = batch['assistant_ranges'][b_idx]['eot']
                    
                    print(f"Number of turns: {len(all_start_indices)}")
                    
                    # Process each turn in the conversation
                    for turn_idx in range(len(all_start_indices)):
                        start_idx = all_start_indices[turn_idx].item()
                        end_idx = all_end_indices[turn_idx].item()
                        eot_idx = all_eot_indices[turn_idx].item()
                        
                        print(f"\nProcessing turn {turn_idx + 1}")
                        print(f"Indices - Start: {start_idx}, End: {end_idx}, EOT: {eot_idx}")
                        
                        # Get context up to this turn
                        context = tokenizer.decode(current_input_ids[0][:start_idx], skip_special_tokens=True)
                        print(f"\nContext for this turn:")
                        print(f"{context}...")
                        
                        # Generate response for this turn
                        curr_input_ids = current_input_ids[:, :start_idx]
                        curr_attention_mask = current_attention_mask[:, :start_idx]
                        
                        
                        print('curr_input_ids.shape', curr_input_ids.shape)
                        print('curr_attention_mask.shape', curr_attention_mask.shape)
                        input('eeeee')
                        print("\nGenerating response...")
                        outputs = model.generate(
                            input_ids=curr_input_ids,
                            attention_mask=curr_attention_mask,
                            max_length=args.pad_to_max,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.convert_tokens_to_ids(['<|eot_id|>'])[0],
                        )
                        
                        # Get the generated response and ground truth
                        generated_response = tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=False)
                        ground_truth = tokenizer.decode(current_input_ids[0][start_idx:eot_idx+1], skip_special_tokens=False)
                        
                        print("\nGround Truth:")
                        print(f"{ground_truth}...")
                        print("\nGenerated Response:")
                        print(f"{generated_response}...")
                        
                        # Store results
                        gt_answers.append(ground_truth)
                        gen_answers.append(generated_response)
                        questions.append(context)
                        
                        # Update conversation history with generated response for next turn
                        if turn_idx < len(all_start_indices) - 1:
                            next_start = all_start_indices[turn_idx + 1].item()
                            current_input_ids = torch.cat([
                                current_input_ids[:, :start_idx],
                                outputs[0][start_idx:].unsqueeze(0),
                                current_input_ids[:, eot_idx+1:next_start]
                            ], dim=1)
                            current_attention_mask = torch.ones_like(current_input_ids)
                        
                        print(f"\nStored results for turn {turn_idx + 1}")
                
            except Exception as e:
                print('\nError occurred during evaluation:')
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Error location: {e.__traceback__.tb_lineno}")
                gt_answers.append("")
                gen_answers.append("")
                questions.append("")
            
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