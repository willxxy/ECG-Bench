import argparse
import glob
import json
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="Organize and print ML experiment results")
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    return parser.parse_args()

def extract_file_info(file):
    parts = file.split('_')
    rag_used = parts[-2] == 'True'
    rag_k = int(parts[-1].split('.')[0]) if rag_used else None
    is_seed = 'seed' in file
    seed_num = int(file.split('/')[-1].split('_')[1]) if is_seed else None
    return rag_used, rag_k, is_seed, seed_num

def process_seed_data(data):
    averages = data['averages']
    metrics = {}
    for metric, value in averages.items():
        if metric == 'ROUGE':
            metrics[metric] = value['rouge-l']
        elif metric == 'BERTSCORE':
            metrics[metric] = sum(value['hf-f1']) / len(value['hf-f1'])
        else:
            metrics[metric] = value
    return metrics

def collect_results(json_files):
    individual_seeds_no_rag = {}
    statistical_no_rag = {}
    individual_seeds_rag = defaultdict(dict)
    statistical_rag = {}

    for file in json_files:
        rag_used, rag_k, is_seed, seed_num = extract_file_info(file)
        with open(file, 'r') as f:
            data = json.load(f)

        if is_seed:
            metrics = process_seed_data(data)
            if rag_used:
                individual_seeds_rag[rag_k][seed_num] = metrics
            else:
                individual_seeds_no_rag[seed_num] = metrics
        else:
            if rag_used:
                statistical_rag[rag_k] = data
            else:
                statistical_no_rag = data

    return (individual_seeds_no_rag, statistical_no_rag,
            individual_seeds_rag, statistical_rag)

def print_seed_results(title, seed_dict):
    if not seed_dict:
        return
    print(title)
    for seed in sorted(seed_dict.keys()):
        print(f"  Seed {seed}:")
        for metric in ['BLEU', 'METEOR', 'ROUGE', 'BERTSCORE', 'ACC']:
            value = seed_dict[seed][metric] * 100  # Scale to 0-100
            print(f"    {metric}: {value:.2f}")
    print('--------------------------------')

def print_statistical_results(title, stats_dict):
    if not stats_dict:
        return
    print(title)
    for metric in ['BLEU', 'METEOR', 'ROUGE', 'BERTSCORE', 'ACC']:
        value = (stats_dict['ROUGE']['rouge-l'] if metric == 'ROUGE' else
                 stats_dict['BERTSCORE']['hf-f1'] if metric == 'BERTSCORE' else
                 stats_dict[metric])
        print(f"  {metric}:")
        for k, v in value.items():
            if k != 'raw_values':
                formatted_v = f"[{v[0]:.2f}, {v[1]:.2f}]" if isinstance(v, list) else f"{v:.2f}"
                print(f"    {k}: {formatted_v}")
    print('--------------------------------')

def main():
    args = get_args()
    dataset_name = args.checkpoint.split('/')[2]
    model_name = args.checkpoint.split('/')[4].split('_')[0]
    
    print(f"Organizing results for {dataset_name} with {model_name}")
    
    json_files = glob.glob(f'{args.checkpoint}/*.json')
    if not json_files:
        print("No results files found.")
        print('================================================')
        return

    (individual_seeds_no_rag, statistical_no_rag,
     individual_seeds_rag, statistical_rag) = collect_results(json_files)

    print_seed_results("Individual Seed Results without RAG:", individual_seeds_no_rag)
    print_statistical_results("Statistical Results without RAG:", statistical_no_rag)
    
    for k in sorted(individual_seeds_rag.keys()):
        print_seed_results(f"Individual Seed Results with RAG k={k}:", individual_seeds_rag[k])
        print_statistical_results(f"Statistical Results with RAG k={k}:", statistical_rag.get(k, {}))

    print('================================================')

if __name__ == '__main__':
    main()