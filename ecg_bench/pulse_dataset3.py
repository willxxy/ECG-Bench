import json
from tqdm import tqdm
import wfdb

def open_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data



# Example usage
file_path = './data/ecg_instruct_pulse/ecg_instruct_pulse.json'
json_data = open_json_file(file_path)
print(type(json_data))
print(len(json_data))
for instance in tqdm(json_data, desc = 'Processing instances'):
    dataset_image_type = instance['image'].split('/')[0]
    
    if dataset_image_type in ('mimic_v4', 'mimic'):
        parts = instance['image'].split('/')
        filename = parts[-1]
        base_filename = filename.split('-')[0]
        path_to_file = '/'.join(parts[1:-1] + [base_filename])
        path_to_dataset = f'./data/mimic/files/{path_to_file}'
        signal, fields = wfdb.rdsamp(path_to_dataset)
        
    elif dataset_image_type == 'ptb-xl':
        parts = instance['image'].split('/')
        filename = parts[-1]
        record_number = filename.split('_')[0]
        subfolder = record_number[:2] + '000'
        path_to_dataset = f'./data/ptb/records500/{subfolder}/{record_number}_hr'
        signal, fields = wfdb.rdsamp(path_to_dataset)
    elif dataset_image_type == 'code15_v4':
        parts = instance['image'].split('/')
        print(parts)
        input()
        