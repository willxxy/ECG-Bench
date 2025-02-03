from datasets import load_dataset

# Create a list of dataset subsets to process
subset_names = ['arena', 'code15-test', 'cpsc-test', 'csn-test-no-cot', 'ecgqa-test', 'g12-test-no-cot', 'mmmu-ecg', 'ptb-test', 'ptb-test-report']

for name in subset_names:
    dataset = load_dataset("PULSE-ECG/ECGBench", name=name, streaming=False, cache_dir='../.huggingface')
    print(dataset['test'][0])
    item = dataset['test'][0]
    print(item['image_path'])
    print(item['image'])
    print(item['conversations'])
    print('--------------------------------')
    input()
    