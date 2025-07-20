from ecg_bench.utils.preprocess_utils import ECGFeatureExtractor
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

class RAGECGDatabase:
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        self.ecg_feature_list = [
                    # "mean",
                    # "std",
                    "max",
                    "min",
                    # "median",
                    # "25th percentile",
                    # "75th percentile",
                    "total power",
                    "peak frequency power",
                    "dominant frequency",
                    "spectral centroid",
                    "heart rate",
                    "heart rate variability",
                    "qrs duration",
                    "t wave amplitude",
                    "st deviation",
                    "wavelet coefficient approximation",
                    "wavelet coefficient detail level 5",
                    "wavelet coefficient detail level 4",
                    "wavelet coefficient detail level 3",
                    "wavelet coefficient detail level 2",
                    "wavelet coefficient detail level 1",
                    "average absolute difference",
                    "root mean square difference"
                ]
    
        
        print('Loading RAG database...')
        if self.args.create_rag_db:
            self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
            self.feature_extractor = ECGFeatureExtractor(self.args.target_sf)
            self.feature_dim = 12* len(self.ecg_feature_list)
            self.signal_dim = 12*self.args.seg_len
            self.feature_weight=np.sqrt(self.signal_dim/self.feature_dim)
            print('Creating RAG database...')
            print('No RAG database found. Creating new one...')
            self.metadata = self.create_and_save_db()
        elif self.args.load_rag_db != None and self.args.load_rag_db_idx != None:
            print('Loading RAG database from file...')
            self.metadata = self.fm.open_json(self.args.load_rag_db)
            self.feature_extractor = ECGFeatureExtractor(self.args.target_sf)
            self.feature_dim = 12*len(self.ecg_feature_list)
            self.signal_dim = 12*self.args.seg_len
            self.feature_weight=np.sqrt(self.signal_dim/self.feature_dim)
            
            if self.args.retrieval_base == 'signal':
                self.signal_index = faiss.read_index(self.args.load_rag_db_idx)         
            elif self.args.retrieval_base == 'feature':
                self.feature_index = faiss.read_index(self.args.load_rag_db_idx)
            elif self.args.retrieval_base == 'combined':
                self.combined_index = faiss.read_index(self.args.load_rag_db_idx)
            else:
                raise ValueError("Please provide a valid retrieval base.")
        else:
            print('Please either create a RAG datbse or load one in.')
        
        print('Metadata loaded.')
        self.reports = [item['report'] for item in self.metadata]
        self.file_paths = [item['file_path'] for item in self.metadata]
        print(f'RAG {self.args.retrieval_base} database loaded.')

        # print('Building sub-indices...')
        # self._build_sub_indices()
        
        print('features dim:', self.feature_dim)
        print('signals dim:', self.signal_dim)
        print('total samples:', len(self.reports))
        print(f'Normalization enabled: {self.args.normalized_rag_feature}')
        print(f'{self.args.retrieval_base} Index loaded.')
    
    def query_signal_lead_normalization(self, signal):
        """
        Normalize each lead individually using z-score normalization.
        """
        if signal.shape[0] == 12: 
            signal = signal.T
            transpose_back = True
        else:
            transpose_back = False
        
        normalized_signal = np.zeros_like(signal, dtype=np.float32)

        for lead_idx in range(12):
            lead_signal = signal[:, lead_idx]
            lead_mean = np.mean(lead_signal)
            lead_std = np.std(lead_signal) + 1e-10
            normalized_signal[:, lead_idx] = (lead_signal - lead_mean) / lead_std

        if transpose_back:
            normalized_signal = normalized_signal.T
    
        return normalized_signal
    
    def query_feature_normalization(self, rag_features):
        """
        Normalize RAG features using z-score normalization.
        """
        expected_total_features = self.feature_dim
        
        if rag_features.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {rag_features.shape}")
        
        if len(rag_features) != expected_total_features:
            raise ValueError(f"Expected {expected_total_features} features for 12-lead ECG, got {len(rag_features)}")
        
        normalized_features = np.zeros_like(rag_features, dtype=np.float32)
        
        for feature_idx, feature_name in enumerate(self.ecg_feature_list):
            feature_values = []
            for lead_idx in range(12):
                feature_pos = lead_idx * len(self.ecg_feature_list) + feature_idx
                feature_values.append(rag_features[feature_pos])
            
            feature_values = np.array(feature_values)
            
            feature_mean = np.mean(feature_values)
            feature_std = np.std(feature_values) + 1e-10 
            
            for lead_idx in range(12):
                feature_pos = lead_idx * len(self.ecg_feature_list) + feature_idx
                normalized_features[feature_pos] = (rag_features[feature_pos] - feature_mean) / feature_std
        
        return normalized_features

    def _build_sub_indices(self):
            ntotal = self.combined_index.ntotal
            nlist=min(100, max(1, ntotal // 30))
            
            feature_vectors = np.zeros((ntotal, self.feature_dim), dtype=np.float32)
            signal_vectors = np.zeros((ntotal, self.signal_dim), dtype=np.float32)

            for i in range(ntotal):
                full_vector = self.combined_index.reconstruct(i)
                feature_vectors[i] = full_vector[:self.feature_dim]
                signal_vectors[i] = full_vector[self.feature_dim:]

            # Build feature index
            quantizer_feature = faiss.IndexFlatL2(self.feature_dim)
            self.feature_index = faiss.IndexIVFFlat(quantizer_feature, self.feature_dim, nlist)
            self.feature_index.train(feature_vectors)
            self.feature_index.add(feature_vectors)
            self.feature_mapping = np.arange(ntotal)

            # Build signal index
            quantizer_signal = faiss.IndexFlatL2(self.signal_dim)
            self.signal_index = faiss.IndexIVFFlat(quantizer_signal, self.signal_dim, nlist)
            self.signal_index.train(signal_vectors)
            self.signal_index.add(signal_vectors)
            self.signal_mapping = np.arange(ntotal)
            
    def create_and_save_db(self):
        print('Initializing RAG database creation...')
        metadata = []
        combined_vectors = []
        feature_vectors = []
        signal_vectors = []
    
        npy_files = list(Path(self.preprocessed_dir).glob('*.npy'))
        if self.args.dev:
            npy_files = npy_files[:300]
            print(f'Development mode: Processing {len(npy_files)} files')
        if self.args.toy:
            npy_files = npy_files[:400000]
            print(f'Toy mode: Processing {len(npy_files)} files')
        
        print(f'Found {len(npy_files)} files to process')
        print(f'Normalization enabled: {self.args.normalized_rag_feature}')
        print('Starting feature extraction from ECG signals...')
            
        for file_path in tqdm(npy_files, desc="Extracting features"):
            try:
                data = self.fm.open_npy(file_path)
                ecg = data['ecg']
                report = data['report']
                features = self.feature_extractor.extract_rag_features(ecg).flatten()
                metadata.append({
                    'report': report,
                    'file_path': str(file_path),
                })

                if not self.args.normalized_rag_feature:
                    signal_vector = ecg.flatten()
                    feature_vector = features.flatten()
                    
                else:
                    signal_vector = self.query_signal_lead_normalization(ecg).flatten()
                    feature_vector = self.query_feature_normalization(features).flatten()

                combined_vector = np.hstack([feature_vector*self.feature_weight, signal_vector])
                signal_vectors.append(signal_vector)
                feature_vectors.append(feature_vector)
                combined_vectors.append(combined_vector)


            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        print(f'Successfully processed {len(metadata)} files')
 
        # Convert to arrays
        feature_array = np.stack(feature_vectors)
        signal_array = np.stack(signal_vectors)
        combined_array = np.stack(combined_vectors)

        # Calculate optimal number of clusters based on dataset size
        ntotal = len(combined_array)
        nlist = min(100, max(1, ntotal // 30))
        
        print(f'Creating FAISS indices with {nlist} clusters for {ntotal} samples...')

        # Create and save feature index
        print('Creating feature index...')
        quantizer_feature = faiss.IndexFlatL2(feature_array.shape[1])
        self.feature_index = faiss.IndexIVFFlat(quantizer_feature, feature_array.shape[1], nlist)
        print('Training feature index...')
        self.feature_index.train(feature_array)
        print('Adding vectors to feature index...')
        self.feature_index.add(feature_array)
        self.feature_index.make_direct_map()
        feature_path = f"./data/{self.args.base_data}/feature_{'normalized' if self.args.normalized_rag_feature else 'unnormalized'}.index"
        print(f'Saving feature index to {feature_path}...')
        faiss.write_index(self.feature_index, feature_path)
        print('Feature index saved successfully!')
        
        # Create and save signal index
        print('Creating signal index...')
        quantizer_signal = faiss.IndexFlatL2(signal_array.shape[1])
        self.signal_index = faiss.IndexIVFFlat(quantizer_signal, signal_array.shape[1], nlist)
        print('Training signal index...')
        self.signal_index.train(signal_array)
        print('Adding vectors to signal index...')
        self.signal_index.add(signal_array)
        self.signal_index.make_direct_map()
        signal_path = f"./data/{self.args.base_data}/signal_{'normalized' if self.args.normalized_rag_feature else 'unnormalized'}.index"
        print(f'Saving signal index to {signal_path}...')
        faiss.write_index(self.signal_index, signal_path)
        print('Signal index saved successfully!')
        
        # Create and save combined index
        print('Creating combined index...')
        quantizer_combined = faiss.IndexFlatL2(combined_array.shape[1])
        self.combined_index = faiss.IndexIVFFlat(quantizer_combined, combined_array.shape[1], nlist)
        print('Training combined index...')
        self.combined_index.train(combined_array)
        print('Adding vectors to combined index...')
        self.combined_index.add(combined_array)
        self.combined_index.make_direct_map()
        combined_path = f"./data/{self.args.base_data}/combined_{'normalized' if self.args.normalized_rag_feature else 'unnormalized'}.index"
        print(f'Saving combined index to {combined_path}...')
        faiss.write_index(self.combined_index, combined_path)
        print('Combined index saved successfully!')
        
        # Save metadata JSON
        metadata_path = f'./data/{self.args.base_data}/rag_metadata.json'
        print(f'Saving metadata to {metadata_path}...')
        self.fm.save_json(metadata, metadata_path)
        print('Metadata saved successfully!')
        
        
        print('RAG database creation completed successfully!')
        print(f'Total samples: {len(metadata)}')
        print(f'Feature dimension: {feature_array.shape[1]}')
        print(f'Signal dimension: {signal_array.shape[1]}')
        print(f'Combined dimension: {combined_array.shape[1]}')
        
        return metadata

    def search_similar(self, query_features=None, query_signal=None, k=5, mode='signal',nprobe=10):
        
        if mode not in ['feature', 'signal', 'combined']:
            raise ValueError("Mode must be 'feature', 'signal', or 'combined'")
            
        if mode == 'feature' and query_features is None:
            raise ValueError("Feature mode requires query_features")
        if mode == 'signal' and query_signal is None:
            raise ValueError("Signal mode requires query_signal")
        if mode == 'combined' and (query_features is None or query_signal is None):
            raise ValueError("Combined mode requires both query_features and query_signal")

        # Set nprobe only for the index that will be used
        if mode == 'feature':
            self.feature_index.nprobe = nprobe
            query_features = query_features.reshape(1, self.feature_dim)
            distances, indices = self.feature_index.search(query_features, k)
            original_indices = indices[0]
        elif mode == 'signal':
            self.signal_index.nprobe = nprobe
            query_signal = query_signal.reshape(1, -1)
            distances, indices = self.signal_index.search(query_signal, k)
            original_indices = indices[0]
            
        else:  # combined mode
            self.combined_index.nprobe = nprobe
            query_features = query_features.reshape(1, self.feature_dim)
            query_signal = query_signal.reshape(1, -1)
            query_combined = np.hstack([query_features*self.feature_weight, query_signal]).reshape(1, -1)

            print(f"Query combined shape: {query_combined.shape}")
            print(f"Combined index dimension: {self.combined_index.d}")
            print(f"Combined index total: {self.combined_index.ntotal}")
            print(f"Query combined sample values: {query_combined[0, :5]}")
            distances, indices = self.combined_index.search(query_combined, k)
            original_indices = indices[0]

        # Prepare results using reconstructed vectors from index
        results = {}
        for i, (dist, idx) in enumerate(zip(distances[0], original_indices)):
            file_path = self.file_paths[idx]
            signal=self.fm.open_npy(file_path)['ecg']
            features=self.feature_extractor.extract_rag_features(signal)

            result_dict = {
                'signal': signal,
                'feature': features,
                'report': self.reports[idx],
                'distance': float(dist),
                'file_path': file_path
            }
            results[i] = result_dict
        
        return results
    
    def format_search(self, results, retrieved_information='combined'):
        if retrieved_information not in ['feature', 'report', 'combined']:
            raise ValueError("retrieved_information must be 'feature', 'report', or 'combined'")
        # results = self.filter_results(results)
        output = f"The following is the top {len(results)} retrieved ECGs and their corresponding "
        
        # Adjust the description based on retrieved_information
        if retrieved_information == 'feature':
            output += "features. Utilize this information to further enhance your response.\n\nThe lead order is I, II, III, aVL, aVR, aVF, V1, V2, V3, V4, V5, V6.\n\n"
        elif retrieved_information == 'report':
            output += "diagnosis. Utilize this information to further enhance your response. \n\n"
        else:  # combined
            output += "features and diagnosis. Utilize this information to further enhance your response. The lead order is I, II, III, aVL, aVR, aVF, V1, V2, V3, V4, V5, V6.\n\n"
        
        for idx, res in results.items():
            # Filter out entries where all feature values are zero
            if np.all(np.array(res['feature']) == 0):
                continue
            
            output += f"Retrieved ECG {idx+1}\n"

            if self.args.dev:
                output+=f"Distance: {res['distance']}\n"
            # Include feature information based on retrieved_information
            if retrieved_information in ['feature', 'combined']:
                output += "Feature Information:\n"
                
                # Organize features by feature type across all leads
                for feature_idx, feature_name in enumerate(self.ecg_feature_list):
                    feature_values = []
                    for lead_idx in range(12):
                        feature_pos = lead_idx * len(self.ecg_feature_list) + feature_idx
                        feature_values.append(round(float(res['feature'][feature_pos]), 6))
                    output += f"{feature_name}: {feature_values}\n"
                output += "\n"

            # Include diagnosis information based on retrieved_information
            if retrieved_information in ['report', 'combined']:
                output += "Diagnosis Information:\n"
                output += f"{res['report']}\n\n"
        return output
    
    def convert_features_to_structured(self, feature_array):
            """
            Convert a flat feature array into a formatted string organized by feature type.
            
            Args:
                feature_array: numpy array of shape (228,) containing RAG features for 12 leads
                
            Returns:
                formatted_string: formatted string with feature names and arrays of 12 values
            """
            if len(feature_array) != self.feature_dim:
                raise ValueError(f"Expected {self.feature_dim} features, got {len(feature_array)}")
            
            formatted_output = ""
            
            for feature_idx, feature_name in enumerate(self.ecg_feature_list):
                feature_values = []
                for lead_idx in range(12):
                    feature_pos = lead_idx * len(self.ecg_feature_list) + feature_idx
                    feature_values.append(round(float(feature_array[feature_pos]), 6))
                formatted_output += f"{feature_name}: {feature_values}\n"
            
            return formatted_output
        
    def filter_results(self, results):
        filtered_results = {}
        count = 0
        for idx, res in results.items():
            feature_array = np.array(res['feature'])
            zero_percentage = np.sum(np.abs(feature_array) < 1e-3) / len(feature_array)
            # total_magnitude = np.sum(np.abs(feature_array))

            if zero_percentage > 0.6:
                continue
            
            filtered_results[count] = res
            count += 1
        return filtered_results

    def test_search(self):
        self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        rng = np.random.RandomState(42)
        npy_files = list(Path(self.preprocessed_dir).glob('*.npy'))
        random_idx = rng.randint(0, len(npy_files))
        query_signal = self.fm.open_npy(npy_files[random_idx])['ecg']
        query_report = self.fm.open_npy(npy_files[random_idx])['report']
        print('query_report: /n', query_report)
        print('query_signal', query_signal.shape)
        
        # Flatten the signal to match the expected dimensions
        query_signal_flat = query_signal.flatten()
        
        start_time = time.time()
        
        # Use retrieval_base parameter to determine search mode
        retrieval_base = getattr(self.args, 'retrieval_base', 'combined')
        if retrieval_base == 'feature':
            # Extract features for feature-based search
            features = self.feature_extractor.extract_rag_features(query_signal)
            if self.args.normalized_rag_feature:
                features = self.query_feature_normalization(features)
            results = self.search_similar(query_features=features, k=3, mode='feature')
        elif retrieval_base == 'combined':
            # Extract features for combined search
            features = self.feature_extractor.extract_rag_features(query_signal)
            if self.args.normalized_rag_feature:
                features = self.query_feature_normalization(features)
                query_signal_flat = self.query_signal_lead_normalization(query_signal).flatten()
            results = self.search_similar(query_features=features, query_signal=query_signal_flat, k=3, mode='combined')
        else:  # signal mode (default)
            if self.args.normalized_rag_feature:
                query_signal_flat = self.query_signal_lead_normalization(query_signal).flatten()
            results = self.search_similar(query_signal=query_signal_flat, k=3, mode='signal')
            
        formatted_results = self.format_search(results, retrieved_information=getattr(self.args, 'retrieved_information', 'combined'))
        print(formatted_results)
        end_time = time.time()
        print(f"Search time: {end_time - start_time:.2f} seconds")