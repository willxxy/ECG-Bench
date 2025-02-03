import json
from tqdm import tqdm
import wfdb
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import threading

def open_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def process_instance(instance):
    try:
        dataset_image_type = instance['image'].split('/')[0]
        
        if dataset_image_type in ('mimic_v4', 'mimic'):
            parts = instance['image'].split('/')
            filename = parts[-1]
            base_filename = filename.split('-')[0]
            path_to_file = '/'.join(parts[1:-1] + [base_filename])
            path_to_dataset = f'./data/mimic/files/{path_to_file}'
            signal, fields = wfdb.rdsamp(path_to_dataset)
            return 'mimic'
            
        elif dataset_image_type == 'ptb-xl':
            parts = instance['image'].split('/')
            filename = parts[-1]
            record_number = filename.split('_')[0]
            subfolder = record_number[:2] + '000'
            path_to_dataset = f'./data/ptb/records500/{subfolder}/{record_number}_hr'
            signal, fields = wfdb.rdsamp(path_to_dataset)
            return 'ptb'
            
    except Exception as e:
        return 'error'

# Example usage
file_path = './data/ecg_instruct_pulse/ecg_instruct_pulse.json'
json_data = open_json_file(file_path)
print(type(json_data))
print(len(json_data))

# Create a thread-safe counter
counter = Counter()
counter_lock = threading.Lock()

# Process files in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    # Use tqdm to show progress
    results = list(tqdm(
        executor.map(process_instance, json_data),
        total=len(json_data)
    ))

# Count the results
counter = Counter(results)

print(f"Good count PTB: {counter['ptb']}")
print(f"Good count MIMIC: {counter['mimic']}")
print(f"Errors: {counter['error']}")
