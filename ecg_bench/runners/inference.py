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
                gt_input_ids = batch['input_ids']
                gt_attention_mask = batch['attn_mask']
                chat_input_ids = gt_input_ids.clone()
                chat_attention_mask = gt_attention_mask.clone()
                assistant_ranges = batch['assistant_ranges']
                offset = 0
                for conv_turn in assistant_ranges:
                    start = conv_turn['start'] + 4 + offset
                    end = conv_turn['end'] + 1 + offset
                    curr_input_ids = chat_input_ids[:, :start]
                    curr_attention_mask = chat_attention_mask[:, :start]
                    # print('curr_input_ids', tokenizer.decode(curr_input_ids[0]))
                    out = model.generate_chat(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        tokenizer=tokenizer
                    )
                    chat_input_ids = torch.cat([
                        chat_input_ids[:, :start],
                        out[:, start:].cpu(),
                        gt_input_ids[:, end-offset:]
                    ], dim=1)
                    # print('chat_input_ids', tokenizer.decode(chat_input_ids[0]))
                    chat_attention_mask = torch.ones_like(chat_input_ids)
                    decoded_out = tokenizer.batch_decode(out[:, start:], skip_special_tokens=False)[0]
                    gt_out = tokenizer.batch_decode(gt_input_ids[:, start-offset:end-offset], skip_special_tokens=False)[0]
                    gt_answers.append(gt_out)
                    gen_answers.append(decoded_out)
                    offset += out[:, start:].size(1) - (end - start)
                    # print('decoded_out', decoded_out)
                    # print('gt_out', gt_out)
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