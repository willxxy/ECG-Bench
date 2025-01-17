import torch
from tqdm import tqdm

def tester(model, dataloader, tokenizer, args, train_utils):
    model.eval()
    len_of_batch = 0
    dev_count = 0
    all_results = []
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
                all_results.append(train_utils.evaluate_strings(answer, out, args.device))
                gt_answers.append(answer[0])
                gen_answers.append(out[0])
                questions.append(batch['question'][0])
            except Exception as e:
                print('could not evaluate for some reason:', str(e))  # prints the error message
                print(f"Error type: {type(e).__name__}")
                print(out)
                print(answer)
                all_results.append({'BLEU': 0, 
                                    'METEOR': 0.0, 
                                    'ROUGE': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}, 
                                    'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]}})
            
            len_of_batch += 1
            
            if args.dev:
                dev_count += 1
                if dev_count == 10:
                    break
                
    # Calculate metrics for this seed
    metric_sums = {
        'BLEU': 0, 'METEOR': 0, 
        'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0,
        'hf-prec': 0, 'hf-rec': 0, 'hf-f1': 0
    }
    metric_counts = {k: 0 for k in metric_sums.keys()}

    for entry in all_results:
        for key, value in entry.items():
            if key in ['ROUGE', 'ROUGE-HF', 'BERTSCORE']:
                for sub_key, sub_value in value.items():
                    if key == 'BERTSCORE':
                        metric_sums[sub_key] += sub_value[0]
                    else:
                        metric_sums[sub_key] += sub_value
                    metric_counts[sub_key] += 1
            else:
                metric_sums[key] += value
                metric_counts[key] += 1

    seed_averages = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
    
    return {
        'metrics': seed_averages,
        'qa_results': {
            'questions': questions,
            'gt_answers': gt_answers,
            'gen_answers': gen_answers
        }
    }
    