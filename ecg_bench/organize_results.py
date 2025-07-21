import glob
import json
from collections import defaultdict
from ecg_bench.config import get_args

def extract_file_info(file):
    filename = file.split('/')[-1]
    parts = filename.split('_')
    
    if filename.startswith('seed_'):
        # seed_{seed}_{perturb}_{rag}_{retrieval_base}_{retrieved_information}_{rag_k}_{rag_prompt_mode}_{normalized_rag_feature}.json
        seed_num = int(parts[1])
        perturb = parts[2]
        rag_used = parts[3] == 'True'
        
        if rag_used:
            retrieval_base = parts[4]
            retrieved_information = parts[5]
            rag_k = int(parts[6])
            rag_prompt_mode = parts[7]+parts[8]
            normalized_rag_feature = parts[9].split('.')[0]
        else:
            retrieval_base = retrieved_information = rag_prompt_mode = normalized_rag_feature = None
            rag_k = None
            
        is_seed = True
    else:
        # statistical_results_{perturb}_{rag}_{retrieval_base}_{retrieved_information}_{rag_k}_{rag_prompt_mode}_{normalized_rag_feature}.json
        perturb = parts[2]
        rag_used = parts[3] == 'True'
        
        if rag_used:
            retrieval_base = parts[4]
            retrieved_information = parts[5]
            rag_k = int(parts[6])
            rag_prompt_mode = parts[7]+parts[8]
            normalized_rag_feature = parts[9].split('.')[0]
        else:
            retrieval_base = retrieved_information = rag_prompt_mode = normalized_rag_feature = None
            rag_k = None
            
        is_seed = False
        seed_num = None
    
    return {
        'rag_used': rag_used,
        'rag_k': rag_k,
        'is_seed': is_seed,
        'seed_num': seed_num,
        'perturb': perturb,
        'retrieval_base': retrieval_base,
        'retrieved_information': retrieved_information,
        'rag_prompt_mode': rag_prompt_mode,
        'normalized_rag_feature': normalized_rag_feature
    }

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
    config_info_no_rag = None
    config_info_rag = {}

    for file in json_files:
        info = extract_file_info(file)
        with open(file, 'r') as f:
            data = json.load(f)

        if info['is_seed']:
            metrics = process_seed_data(data)
            if info['rag_used']:
                individual_seeds_rag[info['rag_k']][info['seed_num']] = metrics
                config_info_rag[info['rag_k']] = info
            else:
                individual_seeds_no_rag[info['seed_num']] = metrics
                config_info_no_rag = info
        else:
            if info['rag_used']:
                statistical_rag[info['rag_k']] = data
                config_info_rag[info['rag_k']] = info
            else:
                statistical_no_rag = data
                config_info_no_rag = info

    return (individual_seeds_no_rag, statistical_no_rag,
            individual_seeds_rag, statistical_rag, config_info_no_rag, config_info_rag)

def print_seed_results(title, seed_dict, config_info=None):
    if not seed_dict:
        return
    print(title)
    if config_info:
        print(f"  Config: perturb={config_info['perturb']}, retrieval_base={config_info['retrieval_base']}, retrieved_info={config_info['retrieved_information']}, prompt_mode={config_info['rag_prompt_mode']}, normalized={config_info['normalized_rag_feature']}")
    for seed in sorted(seed_dict.keys()):
        print(f"  Seed {seed}:")
        for metric in ['BLEU', 'METEOR', 'ROUGE', 'BERTSCORE', 'ACC']:
            value = seed_dict[seed][metric] * 100  # Scale to 0-100
            print(f"    {metric}: {value:.2f}")
    print('--------------------------------')

def print_statistical_results(title, stats_dict, config_info=None):
    if not stats_dict:
        return
    print(title)
    if config_info:
        print(f"  Config: perturb={config_info['perturb']}, retrieval_base={config_info['retrieval_base']}, retrieved_info={config_info['retrieved_information']}, prompt_mode={config_info['rag_prompt_mode']}, normalized={config_info['normalized_rag_feature']}")
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
     individual_seeds_rag, statistical_rag, config_info_no_rag, config_info_rag) = collect_results(json_files)

    print_seed_results("Individual Seed Results without RAG:", individual_seeds_no_rag, config_info_no_rag)
    print_statistical_results("Statistical Results without RAG:", statistical_no_rag, config_info_no_rag)
    
    for k in sorted(individual_seeds_rag.keys()):
        config_info = config_info_rag.get(k)
        print_seed_results(f"Individual Seed Results with RAG k={k}:", individual_seeds_rag[k], config_info)
        print_statistical_results(f"Statistical Results with RAG k={k}:", statistical_rag.get(k, {}), config_info)

    print('================================================')

if __name__ == '__main__':
    main()