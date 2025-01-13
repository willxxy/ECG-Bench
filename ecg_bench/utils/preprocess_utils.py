import numpy as np
from pathlib import Path
import glob
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from scipy import signal
import pywt
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PreprocessECG:
    ''' 
    Main class for preprocessing ECG data. In this class, we do the following:
        1. Prepare the PTB and MIMIC dataframes by getting the ECG paths and reports.
        2. Preprocess the ECG data by applying filters, denoising, and segmenting.
        3. Get the percentiles of the preprocessed ECG data.
        4. Implement clustering based sampling for N percent of the preprocessed ECG data to train tokenizer.
        5. Map external datasets (i.e., PTB and MIMIC variant of ECG-QA and Pretrain Mimic and Mimic Instruct from ECG-Chat)
            to preprocessed PTB and MIMIC base datasets.
    '''
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        self.preprocessed_dir = f'./data/{self.args.data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}'
        
        fm.ensure_directory_exists(f'./pngs')
        
        
        if self.args.map_data == None:
            print(f"Preparing {args.data}")
            if fm.ensure_directory_exists(f'./data/{args.data}/{args.data}.csv') == False:
                print(f"The {args.data} dataframe does not exist. Now preparing the dataframe...")
                self.prepare_df()
            self.df = pd.read_csv(f'./data/{args.data}/{args.data}.csv')
            self.df = self.fm.clean_dataframe(self.df)
            if self.args.dev:
                self.df = self.df.iloc[:1000]
            if self.args.toy:
                self.df = self.df.sample(frac=0.25, random_state=42).reset_index(drop=True)
            print(self.df.head())
            print('Dataframe prepared.')
        else:
            self.ecg_folder = Path(self.preprocessed_dir)
            self.available_ecgs = set(f.stem for f in self.ecg_folder.glob('*'))
    
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
                how='inner')
            
            merged_df = merged_df.dropna(subset=report_columns, how='all')
            df = merged_df[['path', 'report']]    
            
        df.to_csv(f'./data/{self.args.data}/{self.args.data}.csv', index=False)
    
    def preprocess_batch(self):
        self.fm.ensure_directory_exists(self.preprocessed_dir)
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
                
                save_path = f'./data/{self.args.data}_percentiles_{self.args.seg_len}_{self.args.target_sf}_{self.args.num_percentiles}.npy'
                np.save(save_path, percentile_dict)
                
                return percentile_dict
                
        except Exception as e:
            print(f"Error in get_percentiles: {str(e)}")
            return None

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
                
    def map_external_datasets(self):
        valid_instances = []
        
        if self.args.map_data in ['ecg_instruct_45k', 'pretrain_mimic']:
            data = self.fm.open_json(f'./data/{self.args.map_data}/{self.args.map_data}.json')
        elif self.args.map_data in ['ecg-qa_mimic-iv-ecg', 'ecg-qa_ptbxl']:
            paraphrased_jsons = glob.glob(f'./data/ecg-qa/output/{self.args.map_data.split("_")[1]}/paraphrased/*/*.json')
            template_jsons = glob.glob(f'./data/ecg-qa/output/{self.args.map_data.split("_")[1]}/template/*/*.json')
            path_to_all_jsons = paraphrased_jsons + template_jsons
            print('Setting up ECG-QA data...')
            data = self.setup_ecg_qa(path_to_all_jsons)
            
        for instance in tqdm(data, desc = 'Mapping external dataset'):
            if self.args.map_data in ['ecg_instruct_45k', 'pretrain_mimic']:
                text = instance['conversations']
                ecg_path = instance['ecg']
                ecg_path = '_'.join(ecg_path.split('/'))
            elif self.args.map_data in ['ecg-qa_mimic-iv-ecg', 'ecg-qa_ptbxl']:
                text = [instance['question_type'], instance['question'], instance['answer']]
                ecg_path = instance['ecg_path'][0]
                ecg_path = '_'.join(ecg_path.split('/')[2:])
                
            for i in range(10):
                if f"{ecg_path}_{i}" in self.available_ecgs:
                    valid_instances.append({
                        'ecg_path' : f"{self.preprocessed_dir}/{ecg_path}_{i}.npy",
                        'text' : text
                    })
        print(f"Total instances for {self.args.map_data}: {len(data)}")
        print(f'Length of available ecgs: {len(self.available_ecgs)}')
        print(f"Valid instances: {len(valid_instances)}")
        self.fm.save_json(valid_instances, f'./data/{self.args.map_data}_mapped.json')
            
    
    ### HELPER FUNCTIONS
    ### Preprocessing functions
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
                
            downsampled_ecg = downsampled_ecg.astype(np.float32)
                
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
                        'ecg': np.transpose(segmented_ecg[j], (1, 0)),
                        'report': segmented_text[j],
                        'path': self.df.iloc[idx]['path'],
                        'orig_sf': sf,
                        'target_sf': self.args.target_sf,
                        'seg_len': self.args.seg_len
                    }
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
                save_path = f"{self.preprocessed_dir}/{'_'.join(self.df.iloc[idx]['path'].split('/'))}_0.npy"
                np.save(save_path, save_dic)
                
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}. Skipping this instance.")
            return None
        
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
    
    
    ### Preparing ptb and mimic dataframes functions
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
    
    ### Sampling helper functions
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
            freqs, psd = signal.welch(lead_signal, fs=self.args.target_sf, nperseg=1024)
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
            
            peaks, _ = signal.find_peaks(lead_signal, height=0.5*np.max(lead_signal), distance=0.2*self.args.target_sf)
            if len(peaks) > 1:
                # Heart rate
                rr_intervals = np.diff(peaks) / self.args.target_sf
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
        window = int(0.1 * self.args.target_sf)  # 100 ms window
        start = max(0, peak - window)
        end = min(len(ecg), peak + window)
        qrs_segment = ecg[start:end]
        return np.sum(np.abs(qrs_segment) > 0.1 * np.max(qrs_segment)) / self.args.target_sf

    def find_t_wave_amplitude(self, ecg, peaks):
        if len(peaks) < 2:
            return 0
        t_wave_region = ecg[peaks[-2]:peaks[-1]]
        return np.max(t_wave_region) - np.min(t_wave_region)

    def find_st_deviation(self, ecg, peaks):
        if len(peaks) < 2:
            return 0
        st_point = peaks[-1] + int(0.08 * self.args.target_sf)  # 80 ms after R peak
        if st_point < len(ecg):
            return ecg[st_point] - ecg[peaks[-1]]
        return 0

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
                features = self.extract_features(ecg)
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

    def find_elbow_point(self, inertias):
        # Simple method to find the elbow point
        diffs = np.diff(inertias)
        elbow_point = np.argmin(diffs) + 2  # +2 because we started from 2 clusters
        return elbow_point

    ### Mapping external datasets functions
    def setup_ecg_qa(self, glob_paths, question_types=['single-verify', 'single-choose', 'single-query']):
        data = []
        for fname in sorted(glob_paths):
            loaded_file = self.fm.open_json(fname)
            filtered_list = [item for item in loaded_file if item['question_type'] in question_types]
            data.extend(filtered_list)
        return data