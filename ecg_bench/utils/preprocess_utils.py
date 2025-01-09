import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from scipy import signal
import pywt
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

class PreprocessECG:
    
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        
        if fm.ensure_directory_exists(f'./data/{args.data}/{args.data}.csv') == False:
            self.prepare_df()
        self.df = pd.read_csv(f'./data/{args.data}/{args.data}.csv')
        self.df = self.fm.clean_dataframe(self.df)
        if self.args.dev:
            self.df = self.df.iloc[:1000]
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
    
    def _process_single_instance(self, idx):
        save_dic = {}
        file_path = f"{self.add_path}/{self.df.iloc[idx]['path']}"
        report = self.df.iloc[idx]['report']
        
        try:
            ecg, sf = self.fm.open_ecg(file_path)
            assert sf == 500 and ecg.shape == (5000, 12)
            
            if self.args.data == 'mimic':
                ecg = self._reorder_indices(ecg)
                
            filtered_ecg = self.advanced_ecg_filter(ecg, fs=sf)
            denoised_ecg = self.wavelet_denoise(filtered_ecg)
            
            if sf != self.args.target_sf:
                downsampled_ecg = self.nsample_ecg(denoised_ecg, orig_fs=sf, target_fs=self.args.target_sf)
            else:
                downsampled_ecg = denoised_ecg
                
            orig_dur = downsampled_ecg.shape[0] / self.args.target_sf
            segmented_ecg, segmented_text = self.segment_ecg(downsampled_ecg, report, seg_len=self.args.seg_len)
            seg_dur = self.args.seg_len / self.args.target_sf
            
            assert len(segmented_text) == segmented_ecg.shape[0]
            self._check_nan_inf(segmented_ecg, 'preprocessing')
            
            if np.any(np.isnan(segmented_ecg)) or np.any(np.isinf(segmented_ecg)):
                print(f"Warning: NaN values detected in {file_path}. Skipping this instance.")
                return None
            
            if orig_dur != seg_dur:
                for j in range(len(segmented_text)):
                    save_dic = {
                        'ecg': segmented_ecg[j],
                        'report': segmented_text[j],
                        'path': self.df.iloc[idx]['path'],
                        'orig_sf': sf,
                        'target_sf': self.args.target_sf,
                        'seg_len': self.args.seg_len
                    }
                    save_path = f"./data/{self.args.data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}/{'_'.join(self.df.iloc[idx]['path'].split('/'))}_{j}.npy"
                    np.save(save_path, save_dic)
            else:
                save_dic = {
                    'ecg': segmented_ecg[0],
                    'report': segmented_text[0],
                    'path': self.df.iloc[idx]['path'],
                    'orig_sf': sf,
                    'target_sf': self.args.target_sf,
                    'seg_len': self.args.seg_len
                }
                save_path = f"./data/{self.args.data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}/{'_'.join(self.df.iloc[idx]['path'].split('/'))}.npy"
                np.save(save_path, save_dic)
                
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}. Skipping this instance.")
            return None

    def preprocess_batch(self):
        self.fm.ensure_directory_exists(f'./data/{self.args.data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}')
        
        if self.args.data == 'mimic':
            self.add_path = './data/mimic'
        elif self.args.data == 'ptb':
            self.add_path = './data/ptb'
        
        skipped_count = 0        
        try:
            with ProcessPoolExecutor(max_workers=self.args.num_cores) as executor:
                futures = [
                    executor.submit(self._process_single_instance, idx)
                    for idx in range(len(self.df))
                ]
                
                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc='Preprocessing ECGs...'):
                    try:
                        result = future.result()
                        if result is None:
                            skipped_count += 1
                    except Exception as e:
                        print(f"Error processing instance: {str(e)}")
                        skipped_count += 1
        
        except Exception as e:
            print(f"Error in preprocess_instance: {str(e)}")
        
        finally:
            print(f"Total instances skipped: {skipped_count}")
            
    def _process_file_chunk(self, args):
        file_paths, samples_per_chunk = args
        values = []
        total_values = 0
        
        for file_path in file_paths:
            try:
                data = np.load(file_path, allow_pickle=True).item()
                ecg_data = data['ecg']
                flat_data = ecg_data.flatten()
                total_values += len(flat_data)
                values.append(flat_data)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
                
        if not values:
            return np.array([])
            
        concatenated = np.concatenate(values)
        
        if len(concatenated) > samples_per_chunk:
            indices = np.random.choice(len(concatenated), 
                                    size=samples_per_chunk, 
                                    replace=False)
            return concatenated[indices]
        return concatenated

    def get_percentiles(self):
        preprocessed_dir = f'./data/{self.args.data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}'
        
        all_files = [os.path.join(preprocessed_dir, f) for f in os.listdir(preprocessed_dir) 
                    if f.endswith('.npy')]
        
        if not all_files:
            raise ValueError(f"No .npy files found in {preprocessed_dir}")

        num_files = len(all_files)
        num_chunks = self.args.num_cores * 4
        chunk_size = max(1, num_files // num_chunks)
        samples_per_chunk = max(1, self.args.num_percentiles // num_chunks)
        
        file_chunks = [all_files[i:i + chunk_size] for i in range(0, num_files, chunk_size)]
        
        chunk_args = [(chunk, samples_per_chunk) for chunk in file_chunks]

        all_values = []
        percentiles_to_compute = [1, 99]
        total_samples = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.args.num_cores) as executor:
                futures = [executor.submit(self._process_file_chunk, args) 
                        for args in chunk_args]
                
                for future in tqdm(as_completed(futures), 
                                total=len(futures), 
                                desc='Processing files for percentiles'):
                    chunk_values = future.result()
                    if len(chunk_values) > 0:
                        all_values.append(chunk_values)
                        total_samples += len(chunk_values)
                    
                    if total_samples >= self.args.num_percentiles:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

            if all_values:
                final_values = np.concatenate(all_values)
                
                if len(final_values) > self.args.num_percentiles:
                    indices = np.random.choice(len(final_values), 
                                            size=self.args.num_percentiles, 
                                            replace=False)
                    final_values = final_values[indices]
                
                final_percentiles = np.percentile(final_values, percentiles_to_compute)
                
                print(f"\nFinal percentiles (calculated from {len(final_values)} samples):")
                percentile_dict = {}
                for p, v in zip(percentiles_to_compute, final_percentiles):
                    percentile_dict[f'p{p}'] = v
                    print(f"{p}th percentile: {v:.2f}")
                
                save_path = f'./data/{self.args.data}_percentiles_{self.args.seg_len}_{self.args.target_sf}_{self.args.num_percentiles}.npy'
                np.save(save_path, percentile_dict)
                
                return percentile_dict
                
        except Exception as e:
            print(f"Error in get_percentiles: {str(e)}")
            return None

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
    
    def nsample_ecg(self, ecg, orig_fs, target_fs):
        num_samples, num_leads = ecg.shape
        duration = num_samples / orig_fs
        t_original = np.linspace(0, duration, num_samples, endpoint=True)
        t_target = np.linspace(0, duration, int(num_samples * target_fs / orig_fs), endpoint=True)
        
        downsampled_data = np.zeros((len(t_target), num_leads))
        for lead in range(num_leads):
            f = interpolate.interp1d(t_original, ecg[:, lead], kind='cubic', bounds_error=False, fill_value="extrapolate")
            downsampled_data[:, lead] = f(t_target)
        return downsampled_data
    
    def segment_ecg(self, ecg, report, seg_len):
        time_length, _ = ecg.shape
        num_segments = time_length // seg_len
        
        ecg_data_segmented = []
        text_data_segmented = []
        
        for i in range(num_segments):
            start_idx = i * seg_len
            end_idx = (i + 1) * seg_len
            ecg_data_segmented.append(ecg[start_idx:end_idx, :])
            text_data_segmented.append(report)
        
        return np.array(ecg_data_segmented), text_data_segmented