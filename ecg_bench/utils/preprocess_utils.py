import numpy as np
from pathlib import Path
import glob
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from scipy import signal
import pywt
import h5py
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from datasets import load_dataset
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

SUPPORTED_BASE_DATASETS = ['ptb', 'mimic', 'code15', 'cpsc', 'csn']
SUPPORTED_MAPPED_DATASETS = ['ecg_bench_pulse', 'ecg_instruct_pulse', 'pretrain_mimic', 'ecg_instruct_45k', 'ecg-qa_ptbxl', 'ecg-qa_mimic-iv-ecg']

class PrepareDF:
    '''
    Main class for preparing all base datas in a easy to understand dataframe format
    '''
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        if self.args.base_data not in SUPPORTED_BASE_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.args.base_data}")
        
    def prepare_df(self):
        print('Preparing dataframe...')
        if self.args.base_data == 'ptb':
            df = self._prepare_ptb_df()
        elif self.args.base_data == 'mimic':
            df = self._prepare_mimic_df()
        elif self.args.base_data == 'code15':
            df = self._prepare_code15_df()
        elif self.args.base_data == 'cpsc':
            df = self._prepare_cpsc_df()
        elif self.args.base_data == 'csn':
            df = self._prepare_csn_df()
        print('Dataframe prepared.')
        print('Saving dataframe...')
        df.to_csv(f'./data/{self.args.base_data}/{self.args.base_data}.csv', index=False)
        
    def get_df(self):
        print('Getting dataframe...')
        df = pd.read_csv(f'./data/{self.args.base_data}/{self.args.base_data}.csv')
        print('Dataframe retrieved.')
        print('Cleaning dataframe...')
        df = self.fm.clean_dataframe(df)
        print('Dataframe cleaned.')
        if self.args.dev:
            print('Dev mode is on. Reducing dataframe size to 1000 instances...')
            df = df.iloc[:1000]
        if self.args.toy:
            print('Toy mode is on. Reducing dataframe size to 60% of original size...')
            df = df.sample(frac=0.60, random_state=42).reset_index(drop=True)
        print('Dataframe retrieved and cleaned.')
        print(df.head())
        print('Number of instances in dataframe:', len(df))
        print('Dataframe prepared.')
        return df
        
    def _prepare_ptb_df(self):
        ptbxl_database = pd.read_csv('./data/ptb/ptbxl_database.csv', index_col='ecg_id')
        scp_statements = pd.read_csv('./data/ptb/scp_statements.csv')
        ptbxl_database = ptbxl_database.rename(columns={'filename_hr': 'path'})
        df = ptbxl_database[['path', 'report']]
        df = self._translate_german_to_english(df)
        return df
    
    def _prepare_mimic_df(self):
        record_list = pd.read_csv('./data/mimic/record_list.csv')
        machine_measurements = pd.read_csv('./data/mimic/machine_measurements.csv')
        report_columns = [f'report_{i}' for i in range(18)]
        machine_measurements['report'] = machine_measurements[report_columns].apply(
            lambda x: ' '.join([str(val) for val in x if pd.notna(val)]), axis=1)
        mm_columns = ['subject_id', 'study_id'] + report_columns + ['report']
        
        merged_df = pd.merge(
            record_list[['subject_id', 'study_id', 'file_name', 'path']],
            machine_measurements[mm_columns],
            on=['subject_id', 'study_id'],
            how='inner')
        
        merged_df = merged_df.dropna(subset=report_columns, how='all')
        df = merged_df[['path', 'report']]    
        return df
    
    def _prepare_code15_df(self):
        exam_mapping = self._build_code15_exam_mapping()
        df = pd.DataFrame([
            {
                'exam_id': exam_id,
                'path': file_path,
                'idx': idx,
                'report': 'placeholder report'  # Empty report column to match other datasets
            } 
            for exam_id, (file_path, idx) in exam_mapping.items()])
        return df
    
    def _prepare_cpsc_df(self):
        hf_dataset = load_dataset("PULSE-ECG/ECGBench", name='cpsc-test', streaming=False, cache_dir='./../.huggingface')
        cpsc_paths = glob.glob('./data/cpsc/*/*/*.hea')
        cpsc_filename_to_path = {os.path.basename(path).split('.')[0]: path.replace('.hea', '') for path in cpsc_paths}
        df = pd.DataFrame([])
        for item in hf_dataset['test']:
            file_path = item['image_path']
            file_name = file_path.split('/')[-1].split('-')[0]
            conversations = item['conversations']
            if file_name in cpsc_filename_to_path:
                new_row = pd.DataFrame({
                    'path': [cpsc_filename_to_path[file_name]],
                    'report': [conversations],
                    'orig_file_name': [file_name]
                })
                df = pd.concat([df, new_row], ignore_index=True)
        return df
    
    def _prepare_csn_df(self):
        hf_dataset = load_dataset("PULSE-ECG/ECGBench", name='csn-test-no-cot', streaming=False, cache_dir='./../.huggingface')
        csn_paths = glob.glob('./data/csn/WFDBRecords/*/*/*.hea')
        csn_filename_to_path = {os.path.basename(path).split('.')[0]: path.replace('.hea', '') for path in csn_paths}
        df = pd.DataFrame([])
        for item in hf_dataset['test']:
            file_path = item['image_path']
            file_name = file_path.split('/')[-1].split('-')[0]
            conversations = item['conversations']
            if file_name in csn_filename_to_path:
                new_row = pd.DataFrame({
                    'path': [csn_filename_to_path[file_name]],
                    'report': [conversations],
                    'orig_file_name': [file_name]
                })
                df = pd.concat([df, new_row], ignore_index=True)
        return df
    
    
    def _build_code15_exam_mapping(self):
        import h5py
        mapping = {}
        for part in range(18):  
            file_path = f'./data/code15/exams_part{part}.hdf5'
            with h5py.File(file_path, 'r') as f:
                exam_ids = f['exam_id'][:]
                for idx, eid in enumerate(exam_ids):
                    if isinstance(eid, bytes):
                        eid = eid.decode('utf-8')
                    eid = str(int(eid))
                    mapping[eid] = (file_path, idx)
        return mapping
    
    def _translate_german_to_english(self, df):
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


class PreprocessBaseECG:
    '''
    Main class for preprocessing all base datas
    '''
    def __init__(self, args, fm, df):
        self.args = args
        self.fm = fm
        self.df = df
        self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.fm.ensure_directory_exists(folder = self.preprocessed_dir)
        
    def preprocess_batch(self):
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
            
    def _process_single_instance(self, idx):
        save_dic = {}
        try:
            if self.args.base_data =='mimic':
                file_path = f"./data/mimic/{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]['report']
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500 and ecg.shape == (5000, 12)
            elif self.args.base_data == 'ptb':
                file_path = f"./data/ptb/{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]['report']
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500 and ecg.shape == (5000, 12)
            elif self.args.base_data == 'code15':
                file_path = f"{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]['report']
                tracing_idx = self.df.iloc[idx]['idx']
                exam_id = self.df.iloc[idx]['exam_id']
                sf = 400 # code15 has a sampling frequency of 400 Hz
                with h5py.File(file_path, 'r') as f:
                    ecg = f['tracings'][tracing_idx]
                assert ecg.shape == (4096, 12) and sf == 400
            elif self.args.base_data == 'cpsc':
                file_path = f"{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]['report']
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500 # cant assert shape because the shape is variable
            elif self.args.base_data == 'csn':
                file_path = f"{self.df.iloc[idx]['path']}"
                report = self.df.iloc[idx]['report']
                ecg, sf = self.fm.open_ecg(file_path)
                assert sf == 500 and ecg.shape == (5000, 12)
            
            if self.args.base_data == 'mimic' or self.args.base_data == 'code15':
                ecg = self._reorder_indices(ecg)
                
            filtered_ecg = self._advanced_ecg_filter(ecg, fs=sf)
            denoised_ecg = self._wavelet_denoise(filtered_ecg)
            
            if sf != self.args.target_sf:
                downsampled_ecg = self._nsample_ecg(denoised_ecg, orig_fs=sf, target_fs=self.args.target_sf)
            else:
                downsampled_ecg = denoised_ecg
                
            downsampled_ecg = downsampled_ecg.astype(np.float32)
                
            orig_dur = downsampled_ecg.shape[0] / self.args.target_sf
            segmented_ecg, segmented_text = self._segment_ecg(downsampled_ecg, report, seg_len=self.args.seg_len)
            seg_dur = self.args.seg_len / self.args.target_sf
            
            assert len(segmented_text) == segmented_ecg.shape[0]
            self._check_nan_inf(segmented_ecg, 'preprocessing')
            
            if np.any(np.isnan(segmented_ecg)) or np.any(np.isinf(segmented_ecg)):
                print(f"Warning: NaN values detected in {file_path}. Skipping this instance.")
                return None
            
            if orig_dur != seg_dur:
                for j in range(len(segmented_text)):
                    save_dic = {
                        'ecg': np.transpose(segmented_ecg[j], (1, 0)),
                        'report': segmented_text[j],
                        'path': self.df.iloc[idx]['path'],
                        'orig_sf': sf,
                        'target_sf': self.args.target_sf,
                        'seg_len': self.args.seg_len
                    }
                    if self.args.base_data == 'code15':
                        save_dic['exam_id'] = exam_id
                        save_dic['tracing_idx'] = tracing_idx
                        save_path = f"{self.preprocessed_dir}/{exam_id}_{j}.npy"
                    else:
                        save_path = f"{self.preprocessed_dir}/{'_'.join(self.df.iloc[idx]['path'].split('/'))}_{j}.npy"
                    np.save(save_path, save_dic)
            else:
                save_dic = {
                    'ecg': np.transpose(segmented_ecg[0], (1, 0)),
                    'report': segmented_text[0],
                    'path': self.df.iloc[idx]['path'],
                    'orig_sf': sf,
                    'target_sf': self.args.target_sf,
                    'seg_len': self.args.seg_len
                }
                if self.args.base_data == 'code15':
                    save_dic['exam_id'] = exam_id
                    save_dic['tracing_idx'] = tracing_idx
                    save_path = f"{self.preprocessed_dir}/{exam_id}_0.npy"
                else:
                    save_path = f"{self.preprocessed_dir}/{'_'.join(self.df.iloc[idx]['path'].split('/'))}_0.npy"
                np.save(save_path, save_dic)
                
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}. Skipping this instance.")
            return None
     
    def _check_nan_inf(self, ecg, step_name):
        if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
            print(f"Warning: NaN or inf values detected after {step_name}")
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        return ecg
    
    def _reorder_indices(self, ecg):
        if self.args.base_data == 'mimic':
            current_order = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        elif self.args.base_data == 'code15':
            current_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        desired_order = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        order_mapping = {lead: index for index, lead in enumerate(current_order)}
        new_indices = [order_mapping[lead] for lead in desired_order]
        return ecg[:, new_indices]

    def _wavelet_denoise(self, ecg, wavelet='db6', level=4, epsilon=1e-10):
        denoised_ecg = np.zeros_like(ecg)
        for i in range(ecg.shape[1]):
            lead_signal = ecg[:, i]
            
            coeffs = pywt.wavedec(lead_signal, wavelet, level=level)
            median_abs = np.median(np.abs(coeffs[-level]))
            if median_abs == 0:
                threshold = 0
            else:
                threshold = median_abs / 0.6745
            
            def safe_threshold(c):
                thresholded = pywt.threshold(c, threshold, mode='soft')
                return np.where(np.isfinite(thresholded) & (np.abs(c) > epsilon), thresholded, 0)
            
            new_coeffs = [coeffs[0]] + [safe_threshold(c) for c in coeffs[1:]]
            reconstructed = pywt.waverec(new_coeffs, wavelet)
            
            # Ensure the reconstructed signal has the same length as the original
            if len(reconstructed) != ecg.shape[0]:
                reconstructed = signal.resample(reconstructed, ecg.shape[0])
            
            denoised_ecg[:, i] = reconstructed
        
        # Replace any remaining NaN or inf values with zeros
        denoised_ecg = np.nan_to_num(denoised_ecg, nan=0.0, posinf=0.0, neginf=0.0)
        return denoised_ecg

    def _advanced_ecg_filter(self, ecg, fs=500, notch_freqs=[50, 60], highcut=100.0):
        filtered_ecg = ecg.copy()
        
        quality_factor = 30.0
        for notch_freq in notch_freqs:
            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
            # Apply filter to each lead separately to avoid shape issues
            for i in range(filtered_ecg.shape[1]):
                filtered_lead = signal.filtfilt(b_notch, a_notch, filtered_ecg[:, i])
                # Ensure the filtered lead has the same length as the original
                if len(filtered_lead) != filtered_ecg.shape[0]:
                    filtered_lead = signal.resample(filtered_lead, filtered_ecg.shape[0])
                filtered_ecg[:, i] = filtered_lead

        lowcut = 0.5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        order = 4

        b_band, a_band = signal.butter(order, [low, high], btype='band')
        # Apply filter to each lead separately
        for i in range(filtered_ecg.shape[1]):
            filtered_lead = signal.filtfilt(b_band, a_band, filtered_ecg[:, i])
            # Ensure the filtered lead has the same length as the original
            if len(filtered_lead) != filtered_ecg.shape[0]:
                filtered_lead = signal.resample(filtered_lead, filtered_ecg.shape[0])
            filtered_ecg[:, i] = filtered_lead

        baseline_cutoff = 0.05
        baseline_low = baseline_cutoff / nyquist
        b_baseline, a_baseline = signal.butter(order, baseline_low, btype='high')
        # Apply filter to each lead separately
        for i in range(filtered_ecg.shape[1]):
            filtered_lead = signal.filtfilt(b_baseline, a_baseline, filtered_ecg[:, i])
            # Ensure the filtered lead has the same length as the original
            if len(filtered_lead) != filtered_ecg.shape[0]:
                filtered_lead = signal.resample(filtered_lead, filtered_ecg.shape[0])
            filtered_ecg[:, i] = filtered_lead

        return filtered_ecg
    
    def _nsample_ecg(self, ecg, orig_fs, target_fs):
        num_samples, num_leads = ecg.shape
        duration = num_samples / orig_fs
        t_original = np.linspace(0, duration, num_samples, endpoint=True)
        t_target = np.linspace(0, duration, int(num_samples * target_fs / orig_fs), endpoint=True)
        
        downsampled_data = np.zeros((len(t_target), num_leads))
        for lead in range(num_leads):
            f = interpolate.interp1d(t_original, ecg[:, lead], kind='cubic', bounds_error=False, fill_value="extrapolate")
            downsampled_data[:, lead] = f(t_target)
        return downsampled_data
    
    def _segment_ecg(self, ecg, report, seg_len):
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

class SampleBaseECG:
    '''
    Main class for sampling base datas for percentiles and sapmles for tokenizer training
    '''
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.feature_extractor = ECGFeatureExtractor(self.args.target_sf)
    
    def get_percentiles(self):
        all_files = [os.path.join(self.preprocessed_dir, f) for f in os.listdir(self.preprocessed_dir) 
                    if f.endswith('.npy')]
        if not all_files:
            raise ValueError(f"No .npy files found in {self.preprocessed_dir}")
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
                
                save_path = f'./data/{self.args.base_data}_percentiles_{self.args.seg_len}_{self.args.target_sf}_{self.args.num_percentiles}.npy'
                np.save(save_path, percentile_dict)
                
                return percentile_dict
                
        except Exception as e:
            print(f"Error in get_percentiles: {str(e)}")
            return None

    def random_sampling(self):
        # Get all .npy file paths directly using glob
        print("Collecting ECG files for random sampling...")
        file_paths = glob.glob(os.path.join(self.preprocessed_dir, "*.npy"))
        
        # Randomly sample the specified number of files
        print(f"Randomly sampling {self.args.num_tok_samples} files from {len(file_paths)} total files...")
        sampled_files = random.sample(
            file_paths, 
            min(self.args.num_tok_samples, len(file_paths))
        )
        
        # Save the sampled file paths
        save_path = f'./data/sampled_{self.args.num_tok_samples}_random.txt'
        print(f"Sampled {len(sampled_files)} files.")
        with open(save_path, "w") as f:
            for file in sampled_files:
                f.write(f"{file}\n")

    def stratified_sampling(self):
        file_paths, clusters, n_clusters = self.analyze_morphologies()
        
        print(f"Optimal number of clusters: {n_clusters}")
        print("Performing stratified sampling...")
        
        unique_clusters = np.unique(clusters)
        samples_per_cluster = self.args.num_tok_samples // len(unique_clusters)
        
        sampled_files = []
        for cluster in tqdm(unique_clusters, desc="Sampling from clusters"):
            cluster_files = [file_paths[i] for i in range(len(file_paths)) if clusters[i] == cluster]
            sampled_files.extend(random.sample(cluster_files, min(samples_per_cluster, len(cluster_files))))
        
        # If we haven't reached n_samples, randomly sample from the remaining files
        remaining_samples = self.args.num_tok_samples - len(sampled_files)
        if remaining_samples > 0:
            remaining_files = list(set(file_paths) - set(sampled_files))
            sampled_files.extend(random.sample(remaining_files, min(remaining_samples, len(remaining_files))))
        
        save_path = f'./data/sampled_{self.args.num_tok_samples}_{self.args.max_clusters}.txt'
        print(f"Sampled {len(sampled_files)} files.")
        with open(save_path, "w") as f:
            for file in sampled_files:
                f.write(f"{file}\n")
                
    def analyze_morphologies(self):
        
        all_features = []
        file_paths = []

        print("Loading and extracting features from ECG files...")
        count = 0
        for filename in tqdm(os.listdir(self.preprocessed_dir), desc="Extracting features"):
            if filename.endswith('.npy'):
                file_path = os.path.join(self.preprocessed_dir, filename)
                file_paths.append(file_path)
                ecg = self.fm.open_npy(file_path)['ecg']
                features = self.feature_extractor.extract_features(ecg)
                all_features.append(features)
                if count == self.args.num_tok_samples:
                    break
                count+=1

        all_features = np.array(all_features)

        # Perform PCA
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        features_pca = pca.fit_transform(all_features)
        del all_features

        # Scale after PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_pca)
        del features_pca
        
        # Determine optimal number of clusters
        n_clusters = self.find_optimal_clusters(features_scaled, self.args.max_clusters)

        print(f"Optimal number of clusters determined: {n_clusters}")
        print("Clustering all data...")

        # Use KMeans for final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)

        # Try DBSCAN if KMeans results are not satisfactory
        if len(np.unique(clusters)) < 3:
            print("KMeans produced too few clusters. Trying DBSCAN...")
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(features_scaled)

        return file_paths, clusters, len(np.unique(clusters))

    def find_optimal_clusters(self, data, max_clusters):
        inertias = []
        silhouette_scores = []

        for k in tqdm(range(2, max_clusters + 1), desc="Finding optimal clusters"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_, sample_size=10000))

        # Plot elbow curve
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), inertias, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        # Plot silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')

        plt.tight_layout()
        plt.savefig('./pngs/cluster_analysis.png')
        plt.close()

        # Find the elbow point
        elbow_point = self.find_elbow_point(inertias)

        # Find the maximum silhouette score
        max_silhouette = max(silhouette_scores)
        max_silhouette_clusters = silhouette_scores.index(max_silhouette) + 2

        print(f"Elbow method suggests {elbow_point} clusters")
        print(f"Highest silhouette score at {max_silhouette_clusters} clusters")

        # Choose the smaller of the two as a conservative estimate
        optimal_clusters = min(elbow_point, max_silhouette_clusters)
        print(f"Chosen number of clusters: {optimal_clusters}")

        return optimal_clusters

    def _process_file_chunk(self, args):
        file_paths, samples_per_chunk = args
        values = []
        total_values = 0
        
        for file_path in file_paths:
            try:
                data = self.fm.open_npy(file_path)
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

    def find_elbow_point(self, inertias):
        # Simple method to find the elbow point
        diffs = np.diff(inertias)
        elbow_point = np.argmin(diffs) + 2  # +2 because we started from 2 clusters
        return elbow_point

class PreprocessMapECG:
    '''
    Main class for mapping external datasets to base datas
    '''
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        self.available_ecgs = set()
        if self.args.map_data not in SUPPORTED_MAPPED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.args.map_data}")
    
    def map_data(self):
        if self.args.map_data == 'ecg_bench_pulse':
            data = self._prepare_ecg_bench_pulse()
        elif self.args.map_data == 'ecg_instruct_pulse':
            data = self._prepare_ecg_instruct_pulse()
        elif self.args.map_data == 'pretrain_mimic':
            data = self._prepare_pretrain_mimic()
        elif self.args.map_data == 'ecg_instruct_45k':
            data = self._prepare_ecg_instruct_45k()
        elif self.args.map_data == 'ecg-qa_ptbxl':
            data = self._prepare_ecg_qa_ptb()
        elif self.args.map_data == 'ecg-qa_mimic-iv-ecg':
            data = self._prepare_ecg_qa_mimic()
            
        if self.args.dev:
            data = data[:100]
        
        valid_instances = []
        for instance in tqdm(data, desc='Mapping external dataset'):
            ecg_path, text, name, preprocessed_dir = self._process_mapping_instance(instance)
            for i in range(100):
                if f"{ecg_path}_{i}" in self.available_ecgs:
                    valid_instances.append({
                        'ecg_path': f"{preprocessed_dir}/{ecg_path}_{i}.npy",
                        'text': text,
                        'name': name
                    })
        
        print(f"Total instances for {self.args.map_data}: {len(data)}")
        print(f'Length of available ecgs: {len(self.available_ecgs)}')
        print(f"Valid instances: {len(valid_instances)}")
        self.fm.save_json(valid_instances, f'./data/{self.args.map_data}_mapped_{self.args.seg_len}.json')
    
    def _process_mapping_instance(self, instance):
        name = instance.get('name', '')
        
        if self.args.map_data in ['ecg_instruct_45k', 'pretrain_mimic']:
            text = instance['conversations']
            ecg_path = '_'.join(instance['ecg'].split('/'))
            preprocessed_dir = f"./data/mimic/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            
        elif self.args.map_data == 'ecg_instruct_pulse':
            text = instance['conversations']
            ecg_path, preprocessed_dir = self._get_ecg_instruct_pulse_path(instance)
            
        elif self.args.map_data in ['ecg-qa_mimic-iv-ecg', 'ecg-qa_ptbxl']:
            text = [instance['question_type'], instance['question'], instance['answer']]
            ecg_path = '_'.join(instance['ecg_path'][0].split('/')[2:])
            if self.args.map_data == 'ecg-qa_ptbxl':
                preprocessed_dir = f"./data/ptb/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            else:
                preprocessed_dir = f"./data/mimic/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            
        elif self.args.map_data == 'ecg_bench_pulse':
            text = instance['conversations']
            file_name = instance['file_name']
            name = instance['name']
            ecg_path, preprocessed_dir = self._get_ecg_bench_pulse_path(name, file_name)
        
        return ecg_path, text, name, preprocessed_dir
    
    
    def _prepare_ecg_bench_pulse(self):
        json_path = f'./data/{self.args.map_data}/ecg_bench_pulse_datasets.json'
        if self.fm.ensure_directory_exists(file=json_path):
            data = self.fm.open_json(json_path)
        else:
            data = self._setup_ecg_bench_pulse(json_path)
            
        for dataset in ['ptb', 'code15', 'csn', 'cpsc']:
            preprocessed_dir = f"./data/{dataset}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob('*'))
        return data
    
    def _prepare_ecg_instruct_pulse(self):
        for dataset in ['ptb', 'mimic', 'code15']:
            preprocessed_dir = f"./data/{dataset}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob('*'))
            
        data = self.fm.open_json(f'./data/{self.args.map_data}/{self.args.map_data}.json')
        return data
    
    def _prepare_ecg_qa_ptb(self):
        preprocessed_dir = f"./data/ptb/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob('*'))
        dataset_name = self.args.map_data.split("_")[1]
        paraphrased_jsons = glob.glob(f'./data/ecg-qa/output/{dataset_name}/paraphrased/*/*.json')
        template_jsons = glob.glob(f'./data/ecg-qa/output/{dataset_name}/template/*/*.json')
        path_to_all_jsons = paraphrased_jsons + template_jsons
        data = self.setup_ecg_qa(path_to_all_jsons)
        return data
        
    def _prepare_ecg_qa_mimic(self):
        preprocessed_dir = f"./data/mimic/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob('*'))
        dataset_name = self.args.map_data.split("_")[1]
        paraphrased_jsons = glob.glob(f'./data/ecg-qa/output/{dataset_name}/paraphrased/*/*.json')
        template_jsons = glob.glob(f'./data/ecg-qa/output/{dataset_name}/template/*/*.json')
        path_to_all_jsons = paraphrased_jsons + template_jsons
        data = self.setup_ecg_qa(path_to_all_jsons)
        return data
    
    def _prepare_pretrain_mimic(self):
        preprocessed_dir = f"./data/mimic/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob('*'))
        data = self.fm.open_json(f'./data/{self.args.map_data}/{self.args.map_data}.json')
        return data
    
    def _prepare_ecg_instruct_45k(self):
        preprocessed_dir = f"./data/mimic/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.available_ecgs.update(f.stem for f in Path(preprocessed_dir).glob('*'))
        data = self.fm.open_json(f'./data/{self.args.map_data}/{self.args.map_data}.json')
        return data
        
    def _setup_ecg_bench_pulse(self, json_path):
        self.list_of_hf_datasets = ['cpsc-test', 'csn-test-no-cot', 'code15-test', 
                                   'ptb-test', 'ptb-test-report', 'ecgqa-test']
        data = []
        
        for name in tqdm(self.list_of_hf_datasets, desc='Loading ECGBench datasets'):
            dataset = load_dataset("PULSE-ECG/ECGBench", name=name, 
                                  streaming=False, cache_dir='./../.huggingface')
            
            for item in dataset['test']:
                conversations = item['conversations']
                file_path = item['image_path']
                file_name = file_path.split('/')[-1].split('-')[0]
                
                # Handle ecgqa-test special case
                if name == 'ecgqa-test':
                    for conv in conversations:
                        if isinstance(conv.get('value'), list):
                            conv['value'] = ''.join(conv['value'])
                
                data.append({
                    'file_path': file_path,
                    'file_name': file_name,
                    'conversations': conversations,
                    'name': name
                })
        
        self.fm.save_json(data, json_path)
        return data
    
    def _get_ecg_bench_pulse_path(self, name, file_name):
        if name in ['ecgqa-test', 'ptb-test-report', 'ptb-test']:
            preprocessed_dir = f"./data/ptb/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            subfolder = file_name[:2] + '000'
            return f"records500_{subfolder}_{file_name}", preprocessed_dir
        elif name == 'cpsc-test':
            cpsc_paths = glob.glob('./data/cpsc/*/*/*.hea')
            cpsc_filename_to_path = {os.path.basename(path).split('.')[0]: path.replace('.hea', '') 
                                     for path in cpsc_paths}
            preprocessed_dir = f"./data/cpsc/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            ecg_path = cpsc_filename_to_path[file_name]
            return '_'.join(ecg_path.split('/')), preprocessed_dir
        elif name == 'csn-test-no-cot':
            csn_paths = glob.glob('./data/csn/WFDBRecords/*/*/*.hea')
            csn_filename_to_path = {os.path.basename(path).split('.')[0]: path.replace('.hea', '') 
                                    for path in csn_paths}
            preprocessed_dir = f"./data/csn/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            ecg_path = csn_filename_to_path[file_name]
            return '_'.join(ecg_path.split('/')), preprocessed_dir
        elif name == 'code15-test':
            preprocessed_dir = f"./data/code15/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            return file_name.split('-')[0], preprocessed_dir
        
        return ""
    
    def _get_ecg_instruct_pulse_path(self, instance):
        parts = instance['image'].split('/')
        dataset_image_type = parts[0]
        filename = parts[-1]
        
        if dataset_image_type in ['mimic_v4', 'mimic']:
            dataset_image_type = 'mimic'
            base_filename = filename.split('-')[0]
            path_to_file = '_'.join(parts[1:-1] + [base_filename])
            ecg_path = f"files_{path_to_file}"
        elif dataset_image_type in ['ptb-xl']:
            dataset_image_type = 'ptb'
            record_number = filename.split('_')[0]
            record_number = f"{record_number}_hr"
            subfolder = record_number[:2] + '000'
            ecg_path = f"records500_{subfolder}_{record_number}"
        elif dataset_image_type in ['code15_v4']:
            dataset_image_type = 'code15'
            ecg_path = filename.split('-')[0]
        
        preprocessed_dir = f"./data/{dataset_image_type}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        return ecg_path, preprocessed_dir
    
    def setup_ecg_qa(self, glob_paths, question_types=['single-verify', 'single-choose', 'single-query']):
        data = []
        for fname in sorted(glob_paths):
            loaded_file = self.fm.open_json(fname)
            filtered_list = [item for item in loaded_file if item['question_type'] in question_types]
            data.extend(filtered_list)
        return data
    
    
class ECGFeatureExtractor:
    '''
    Class for extracting various features from ECG signals
    '''
    def __init__(self, target_sf):
        self.target_sf = target_sf
        
    def extract_features(self, ecg):
        features = []
        
        for lead in range(ecg.shape[0]):
            lead_signal = ecg[lead, :]
            
            # Basic statistical features
            features.extend([
                np.mean(lead_signal),
                np.std(lead_signal),
                np.max(lead_signal),
                np.min(lead_signal),
                np.median(lead_signal),
                np.percentile(lead_signal, 25),
                np.percentile(lead_signal, 75)
            ])
            
            # Frequency domain features
            freqs, psd = signal.welch(lead_signal, fs=self.target_sf, nperseg=1024)
            total_power = np.sum(psd)
            features.extend([
                total_power,  # Total power
                np.max(psd),  # Peak frequency power
                freqs[np.argmax(psd)],  # Dominant frequency
            ])
            
            # Spectral centroid with NaN handling
            if total_power > 0:
                spectral_centroid = np.sum(freqs * psd) / total_power
            else:
                spectral_centroid = 0  # or another appropriate default value
            features.append(spectral_centroid)
            
            peaks, _ = signal.find_peaks(lead_signal, height=0.5*np.max(lead_signal), distance=0.2*self.target_sf)
            if len(peaks) > 1:
                # Heart rate
                rr_intervals = np.diff(peaks) / self.target_sf
                heart_rate = 60 / np.mean(rr_intervals)
                features.append(heart_rate)
                
                # Heart rate variability
                hrv = np.std(rr_intervals)
                features.append(hrv)
                
                # QRS duration (simplified)
                qrs_duration = np.mean([self.find_qrs_duration(lead_signal, peak) for peak in peaks])
                features.append(qrs_duration)
            else:
                features.extend([0, 0, 0])  # Placeholder values if no peaks found
            
            # T-wave features (simplified)
            t_wave_amp = self.find_t_wave_amplitude(lead_signal, peaks)
            features.append(t_wave_amp)
            
            # ST segment features (simplified)
            st_deviation = self.find_st_deviation(lead_signal, peaks)
            features.append(st_deviation)
            
            coeffs = pywt.wavedec(lead_signal, 'db4', level=5)
            features.extend([np.mean(np.abs(c)) for c in coeffs])
            
            # Non-linear features
            features.append(np.mean(np.abs(np.diff(lead_signal))))  # Average absolute difference
            features.append(np.sqrt(np.mean(np.square(np.diff(lead_signal)))))  # Root mean square of successive differences
        
        return np.array(features)

    def find_qrs_duration(self, ecg, peak):
        # Simplified QRS duration estimation
        window = int(0.1 * self.target_sf)  # 100 ms window
        start = max(0, peak - window)
        end = min(len(ecg), peak + window)
        qrs_segment = ecg[start:end]
        return np.sum(np.abs(qrs_segment) > 0.1 * np.max(qrs_segment)) / self.target_sf

    def find_t_wave_amplitude(self, ecg, peaks):
        if len(peaks) < 2:
            return 0
        t_wave_region = ecg[peaks[-2]:peaks[-1]]
        return np.max(t_wave_region) - np.min(t_wave_region)

    def find_st_deviation(self, ecg, peaks):
        if len(peaks) < 2:
            return 0
        st_point = peaks[-1] + int(0.08 * self.target_sf)  # 80 ms after R peak
        if st_point < len(ecg):
            return ecg[st_point] - ecg[peaks[-1]]
        return 0
        
    