from datasets import load_dataset
import glob
import os
# Create a list of dataset subsets to process
# subset_names = ['arena', 'code15-test', 'cpsc-test', 'csn-test-no-cot', 'ecgqa-test', 'g12-test-no-cot', 'mmmu-ecg', 'ptb-test', 'ptb-test-report']
# to not include mmmu-ecg, arena, g12-test-no-cot
subset_names = ['cpsc-test', 'csn-test-no-cot']

# Extract just the base filenames without extensions for comparison
cpsc_paths = glob.glob('./data/cpsc/*/*/*.hea')
cpsc_filenames = [os.path.basename(path).split('.')[0] for path in cpsc_paths]
print('cpsc', len(cpsc_paths))

csn_paths = glob.glob('./data/csn/WFDBRecords/*/*/*.hea')
csn_filenames = [os.path.basename(path).split('.')[0] for path in csn_paths]
print('csn', len(csn_paths))

num_cpsc = 0
num_csn = 0

for name in subset_names:
    dataset = load_dataset("PULSE-ECG/ECGBench", name=name, streaming=False, cache_dir='../.huggingface')
    print(name)
    print(len(dataset['test']))
    for item in dataset['test']:
        print(item)
        file_path = item['image_path']
        file_name = file_path.split('/')[-1].split('-')[0]
        dataset_name = file_path.split('/')[0]
        conversations = item['conversations']
        print(conversations)
        input()
        if dataset_name == 'csn':
            if file_name in csn_filenames:
                num_csn += 1
        elif dataset_name == 'cpsc':
            if file_name in cpsc_filenames:
                num_cpsc += 1
        
        
        if name == 'ecgqa-test':
            for conv in conversations: # only ecgqa test
                if isinstance(conv.get('value'), list):
                    conv['value'] = ''.join(conv['value'])
                
        # print(conversations)
        # print(file_path)
        # break
    print('--------------------------------')

print('cpsc', num_cpsc)
print('csn', num_csn)
    