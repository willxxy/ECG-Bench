import numpy as np
import pandas as pd

class PreprocessECG:
    
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        if self.args.data == 'ptb':
            ptbxl_database = pd.read_csv('./data/ptb/ptbxl_database.csv', index_col='ecg_id')
            scp_statements = pd.read_csv('./data/ptb/scp_statements.csv')
            ptbxl_database = ptbxl_database.rename(columns={'filename_hr': 'path'})
            self.df = ptbxl_database[['path', 'report']]
            
        elif self.args.data == 'mimic':
            record_list = pd.read_csv('./data/mimic/record_list.csv')
            machine_measurements = pd.read_csv('./data/mimic/machine_measurements.csv')
            waveform_note_links = pd.read_csv('./data/mimic/waveform_note_links.csv')
            
            report_columns = [f'report_{i}' for i in range(18)]
            
            machine_measurements['report'] = machine_measurements[report_columns].apply(
                lambda x: ' '.join([str(val) for val in x if pd.notna(val)]), axis=1)
            
            mm_columns = ['subject_id', 'study_id'] + report_columns + ['report']
            
            merged_df = pd.merge(
                record_list[['subject_id', 'study_id', 'file_name', 'path']],
                machine_measurements[mm_columns],
                on=['subject_id', 'study_id'],
                how='inner'
            )
            merged_df = merged_df.dropna(subset=report_columns, how='all')
            self.df = merged_df[['path', 'report']]
        print(self.df.head())
    
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