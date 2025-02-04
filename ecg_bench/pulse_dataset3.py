import json
from tqdm import tqdm
import wfdb
import h5py

def build_code15_exam_mapping():
    """
    Builds a dictionary mapping exam identifiers to a tuple with the file path
    and the index position inside that HDF5 file.
    
    Returns:
        dict: A dictionary where the key is the exam_id (str) and the value is a tuple (file_path, index_in_file)
    """
    mapping = {}
    total_instances = 0
    all_exam_ids = []
    
    # Loop over exam parts (assuming parts 0 through 17)
    for part in range(18):  
        file_path = f'./data/code15/exams_part{part}.hdf5'
        with h5py.File(file_path, 'r') as f:
            exam_ids = f['exam_id'][:]  # exams in this file
            total_instances += len(f['tracings'])
            all_exam_ids.extend([str(int(eid)) for eid in exam_ids])
            
            # Build mapping for each exam in this file
            for idx, eid in enumerate(exam_ids):
                eid = str(int(eid))
                if isinstance(eid, bytes):
                    eid = eid.decode('utf-8')
                mapping[eid] = (file_path, idx)
    
    print(f"Total instances across all files: {total_instances}")
    print(f"Total exam_ids across all files: {len(all_exam_ids)}")
    print(f"Unique exam_ids across all files: {len(set(all_exam_ids))}")
    
    return mapping

def open_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

code15_exam_mapping = build_code15_exam_mapping()


#Example usage
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
        exam_identifier = parts[-1].split('-')[0]
        file_path, idx = code15_exam_mapping[exam_identifier]
        with h5py.File(file_path, 'r') as f:
            tracing = f['tracings'][idx]
        print(tracing.shape)
        print(instance)
        input()