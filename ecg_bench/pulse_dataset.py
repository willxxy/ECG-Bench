from datasets import load_dataset
import glob
import os
# Create a list of dataset subsets to process
# subset_names = ['arena', 'code15-test', 'cpsc-test', 'csn-test-no-cot', 'ecgqa-test', 'g12-test-no-cot', 'mmmu-ecg', 'ptb-test', 'ptb-test-report']
# to not include mmmu-ecg, arena, g12-test-no-cot
subset_names = ['ecgqa-test', 'ptb-test-report', 'ptb-test']

# ecgqa-test, ptb-test-report, ptb-test goes to ptbxl


for name in subset_names:
    dataset = load_dataset("PULSE-ECG/ECGBench", name=name, streaming=False, cache_dir='../.huggingface')
    print(name)
    print(len(dataset['test']))
    for item in dataset['test']:
        file_path = item['image_path']
        file_name = file_path.split('/')[-1].split('-')[0]
        dataset_name = file_path.split('/')[0]
        conversations = item['conversations']
        if name == 'ecgqa-test':
            for conv in conversations: # only ecgqa test
                if isinstance(conv.get('value'), list):
                    conv['value'] = ''.join(conv['value'])
        print(file_path)
        print(file_name)
        break
                
        # print(conversations)
        # print(file_path)
        # break
    print('--------------------------------')

    