import json
import numpy as np

def open_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


file_path = './data/ecg_instruct_pulse_mapped.json'
json_data = open_json_file(file_path)
print(json_data[0])

print(np.load(json_data[0]['ecg_path'], allow_pickle=True).item()['ecg'].shape)