import torch
from tqdm import tqdm

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