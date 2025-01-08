import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from scipy import signal
import pywt

class PreprocessECG:
    
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        
        if fm.ensure_directory_exists(f'./data/{args.data}/{args.data}.csv') == False:
            self.prepare_df()
        self.df = pd.read_csv(f'./data/{args.data}/{args.data}.csv')
        self.df = self.fm.clean_dataframe(self.df)
        print(self.df.head())
    
    ### MAIN FUNCTIONS
    def prepare_df(self):
        if self.args.data == 'ptb':
            ptbxl_database = pd.read_csv('./data/ptb/ptbxl_database.csv', index_col='ecg_id')
            scp_statements = pd.read_csv('./data/ptb/scp_statements.csv')
            ptbxl_database = ptbxl_database.rename(columns={'filename_hr': 'path'})
            df = ptbxl_database[['path', 'report']]
            df = self.translate_german_to_english(df)
            
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
            df = merged_df[['path', 'report']]
        
        df.to_csv(f'./data/{self.args.data}/{self.args.data}.csv', index=False)
    
    def preprocess(self):
        self.fm.ensure_directory_exists(f'./data/{self.args.data}/preprocessed')
        if self.args.data == 'mimic':
            add_path = './data/mimic'
        elif self.args.data == 'ptb':
            add_path = './data/ptb'
        count = 0
        for i in tqdm(range(len(self.df)), desc = 'Preprocessing ECGs...'):
            file_path = f"{add_path}/{self.df.iloc[i]['path']}"
            report = self.df.iloc[i]['report']
            ecg, sf = self.fm.open_ecg(file_path)
            assert sf == 500 and ecg.shape == (5000, 12)
            if self.args.data == 'mimic':
                ecg = self._reorder_indices(ecg)
            
            print(ecg.shape, sf)
            count +=1
            if count == 5:
                break

    ### HELPER FUNCTIONS
    def _check_nan_inf(self, ecg, step_name):
        if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
            print(f"Warning: NaN or inf values detected after {step_name}")
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        return ecg
    
    def _reorder_indices(self, ecg):
        current_order = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        desired_order = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        order_mapping = {lead: index for index, lead in enumerate(current_order)}
        new_indices = [order_mapping[lead] for lead in desired_order]
        return ecg[:, new_indices]
    
    def translate_german_to_english(self, df):
        texts = df['report'].values
        try:
            if isinstance(texts, list):
                texts = np.array(texts)
            
            if not isinstance(texts, np.ndarray):
                raise ValueError("Input must be a numpy array or list")
            if texts.ndim != 1:
                raise ValueError(f"Expected 1D array, got shape {texts.shape}")
            if len(texts) == 0:
                raise ValueError("Input array cannot be empty")
                
            valid_mask = np.array([bool(text and str(text).strip()) for text in texts])
            valid_texts = texts[valid_mask]
            
            if len(valid_texts) == 0:
                raise ValueError("All input texts are empty")
                
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir='./../.huggingface')
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir='./../.huggingface').to(device)
            
            batch_size = 64
            translations = []
            
            for i in tqdm(range(0, len(valid_texts), batch_size), desc = 'Translating files'):
                batch_texts = valid_texts[i:i + batch_size]
                
                encoded = tokenizer(list(batch_texts), return_tensors="pt", padding=True, truncation=True)
                encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **encoded,
                        max_length=128,
                    )
                
                batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translations.extend(batch_translations)
            
            result = np.empty_like(texts, dtype=object)
            result[valid_mask] = translations
            result[~valid_mask] = ''
            
            translated_df = df.copy()
            translated_df['report'] = result
            
            return translated_df
        
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Translation error: {str(e)}")

    def wavelet_denoise(self, ecg, wavelet='db6', level=4, epsilon=1e-10):
        denoised_ecg = np.zeros_like(ecg)
        for i in range(ecg.shape[1]):
            coeffs = pywt.wavedec(ecg[:, i], wavelet, level=level)
            median_abs = np.median(np.abs(coeffs[-level]))
            if median_abs == 0:
                threshold = 0
            else:
                threshold = median_abs / 0.6745
            
            def safe_threshold(c):
                thresholded = pywt.threshold(c, threshold, mode='soft')
                return np.where(np.isfinite(thresholded) & (np.abs(c) > epsilon), thresholded, 0)
            
            new_coeffs = [coeffs[0]] + [safe_threshold(c) for c in coeffs[1:]]
            denoised_ecg[:, i] = pywt.waverec(new_coeffs, wavelet)
        
        # Replace any remaining NaN or inf values with zeros
        denoised_ecg = np.nan_to_num(denoised_ecg, nan=0.0, posinf=0.0, neginf=0.0)
        return denoised_ecg

    def advanced_ecg_filter(self, ecg, fs=500, notch_freqs=[50, 60], highcut=100.0):
        filtered_ecg = ecg.copy()
        
        quality_factor = 30.0
        for notch_freq in notch_freqs:
            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
            filtered_ecg = signal.filtfilt(b_notch, a_notch, filtered_ecg, axis=0)

        lowcut = 0.5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        order = 4

        b_band, a_band = signal.butter(order, [low, high], btype='band')
        filtered_ecg = signal.filtfilt(b_band, a_band, filtered_ecg, axis=0)

        baseline_cutoff = 0.05
        baseline_low = baseline_cutoff / nyquist
        b_baseline, a_baseline = signal.butter(order, baseline_low, btype='high')
        filtered_ecg = signal.filtfilt(b_baseline, a_baseline, filtered_ecg, axis=0)

        return filtered_ecg