import numpy as np
import pandas as pd

class PreprocessECG:
    
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        if self.args.data == 'ptb':
            ptbxl_database = pd.read_csv('./data/ptb/ptbxl_database.csv', index_col='ecg_id')
            print(ptbxl_database.head())
            print(ptbxl_database.columns)
            scp_statements = pd.read_csv('./data/ptb/scp_statements.csv')
            print(scp_statements.head())
            print(scp_statements.columns)
        elif self.args.data == 'mimic':
            record_list = pd.read_csv('./data/mimic/record_list.csv')
            print(record_list.head())
            print(record_list.columns)
            machine_measurements = pd.read_csv('./data/mimic/machine_measurements.csv')
            print(machine_measurements.head())
            print(machine_measurements.columns)
            waveform_note_links = pd.read_csv('./data/mimic/waveform_note_links.csv')
            print(waveform_note_links.head())
            print(waveform_note_links.columns)
    
    def _check_nan_inf(self, signal, step_name):
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print(f"Warning: NaN or inf values detected after {step_name}")
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        return signal
    
    def _reorder_indices(self, signal):
        current_order = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        desired_order = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        order_mapping = {lead: index for index, lead in enumerate(current_order)}
        new_indices = [order_mapping[lead] for lead in desired_order]
        return signal[:, new_indices]