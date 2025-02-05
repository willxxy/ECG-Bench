from datasets import load_dataset

# Create a list of dataset subsets to process
subset_names = ['arena', 'code15-test', 'cpsc-test', 'csn-test-no-cot', 'ecgqa-test', 'g12-test-no-cot', 'mmmu-ecg', 'ptb-test', 'ptb-test-report']

for name in subset_names:
    dataset = load_dataset("PULSE-ECG/ECGBench", name=name, streaming=False, cache_dir='../.huggingface')
    print(len(dataset['test']))
    # for item in dataset['test']:
    #     file_path = item['image_path']
    #     conversations = item['conversations']
        
    
    