import torch
from tqdm import tqdm
import numpy as np

def tester_chat(model, dataloader, tokenizer, args, train_utils):
    model.eval()
    len_of_batch = 0
    dev_count = 0
    gt_answers = []
    gen_answers = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Testing {args.model}', position=0, leave=True)):
            if batch is None:
                print("Skipping invalid batch")
                continue
            
            try:
                gt_input_ids = batch['input_ids']
                gt_attention_mask = batch['attn_mask']
                chat_input_ids = gt_input_ids.clone()
                chat_attention_mask = gt_attention_mask.clone()
                assistant_ranges = batch['assistant_ranges']
                if args.inference == 'second':
                    encoder_out = batch['encoder_out']
                    signal_id_index = batch['signal_id_index'].item()
                offset = 0
                for conv_turn in assistant_ranges:
                    print('conv_turn', conv_turn)
                    start = conv_turn['start'] + 4 + offset
                    end = conv_turn['end'] + 1 + offset
                    curr_input_ids = chat_input_ids[:, :start]
                    curr_attention_mask = chat_attention_mask[:, :start]
                    if args.dev:
                        print('curr_input_ids', tokenizer.decode(curr_input_ids[0]))
                        print('--------------------------------' * 3)
                    
                    if args.inference == 'second':
                        out = model.generate_chat(
                            input_ids=curr_input_ids,
                            attention_mask=curr_attention_mask,
                            tokenizer=tokenizer,
                            encoder_out=encoder_out,
                            signal_id_index=signal_id_index)
                    else:
                        out = model.generate_chat(
                            input_ids=curr_input_ids,
                            attention_mask=curr_attention_mask,
                            tokenizer=tokenizer)
                    
                    chat_input_ids = torch.cat([
                        chat_input_ids[:, :start],
                        out[:, start:].cpu(),
                        gt_input_ids[:, end-offset:]
                    ], dim=1)
                    chat_attention_mask = torch.ones_like(chat_input_ids)
                    decoded_out = tokenizer.batch_decode(out[:, start:], skip_special_tokens=False)[0]
                    gt_out = tokenizer.batch_decode(gt_input_ids[:, start-offset:end-offset], skip_special_tokens=False)[0]
                    gt_answers.append(gt_out)
                    gen_answers.append(decoded_out)
                    offset += out[:, start:].size(1) - (end - start)
                    if args.dev:
                        print('chat_input_ids', tokenizer.decode(chat_input_ids[0]))
                        print('--------------------------------' * 3)
                        print('decoded_out', decoded_out)
                        print('gt_out', gt_out)
                        print('--------------------------------' * 3)
            except Exception as e:
                print('\nError occurred during evaluation:')
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Error location: {e.__traceback__.tb_lineno}")
                gt_answers.append("")
                gen_answers.append("")
            
            len_of_batch += 1
            
            if args.dev:
                print(f"\nCompleted batch {batch_idx}. Total conversations processed: {len_of_batch}")
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
        print(f"Accuracy: {all_metrics['ACC']}")
    except Exception as e:
        print('\nError during metric calculation:')
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        all_metrics = {
            'BLEU': 0, 
            'METEOR': 0.0, 
            'ROUGE': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}, 
            'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]},
            'ACC': 0.0
        }
    
    print("\nEvaluation complete!")
    return {
        'metrics': all_metrics,
        'qa_results': {
            'gt_answers': gt_answers,
            'gen_answers': gen_answers
        }
    }