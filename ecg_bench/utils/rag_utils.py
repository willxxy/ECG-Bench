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
                    "mean",
                    "std",
                    "max",
                    "min",
                    "median",
                    "25th percentile",
                    "75th percentile",
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
            print('Creating RAG database...')
            print('No RAG database found. Creating new one...')
            self.metadata = self.create_and_save_db()
        elif self.args.load_rag_db != None and self.args.load_rag_db_idx != None:
            print('Loading RAG database from file...')
            self.metadata = self.fm.open_json(self.args.load_rag_db)
            self.index = faiss.read_index(self.args.load_rag_db_idx)
        else:
            print('Please either create a RAG datbse or load one in.')
        
        print('Metadata loaded.')
        self.reports = [item['report'] for item in self.metadata]
        self.file_paths = [item['file_path'] for item in self.metadata]
        
        print('RAG database loaded.')
        # Get dimensions from first vector in index
        first_vector = self.index.reconstruct(0)
        self.feature_dim = 288
        self.signal_dim = len(first_vector) - self.feature_dim
        print('Building sub-indices...')
        self._build_sub_indices()
        
        print('features dim:', self.feature_dim)
        print('signals dim:', self.signal_dim)
        print('total samples:', len(self.reports))
        print('Index loaded.')
    
    def _build_sub_indices(self):
            ntotal = self.index.ntotal
            nlist = min(100, max(1, ntotal // 30))
            
            feature_vectors = np.zeros((ntotal, self.feature_dim), dtype=np.float32)
            signal_vectors = np.zeros((ntotal, self.signal_dim), dtype=np.float32)

            for i in range(ntotal):
                full_vector = self.index.reconstruct(i)
                feature_vectors[i] = full_vector[:self.feature_dim]
                signal_vectors[i] = full_vector[self.feature_dim:]

            # Build feature index
            quantizer_feature = faiss.IndexFlatL2(self.feature_dim)
            self.feature_index = faiss.IndexIVFFlat(quantizer_feature, self.feature_dim, nlist)
            self.feature_index.train(feature_vectors)
            self.feature_index.add(feature_vectors)
            self.index_mapping = np.arange(ntotal)

            # Build signal index
            quantizer_signal = faiss.IndexFlatL2(self.signal_dim)
            self.signal_index = faiss.IndexIVFFlat(quantizer_signal, self.signal_dim, nlist)
            self.signal_index.train(signal_vectors)
            self.signal_index.add(signal_vectors)
            self.signal_mapping = np.arange(ntotal)
            
    def create_and_save_db(self):
        metadata = []
        vectors_for_index = []
        
        
        npy_files = list(Path(self.preprocessed_dir).glob('*.npy'))
        if self.args.dev:
            npy_files = npy_files[:1000]
        if self.args.toy:
            npy_files = npy_files[:400000]
            
        for file_path in tqdm(npy_files, desc="Extracting features"):
            try:
                data = self.fm.open_npy(file_path)
                ecg = data['ecg']
                report = data['report']
                features = self.feature_extractor.extract_features(ecg)
                
                # Store vectors for FAISS index
                combined_vector = np.hstack([features.flatten(), ecg.flatten()])
                vectors_for_index.append(combined_vector)
                
                # Store only metadata in JSON
                metadata.append({
                    'report': report,
                    'file_path': str(file_path)
                })
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
                
        # Create and save index
        vectors_array = np.stack(vectors_for_index)

        # Calculate optimal number of clusters based on dataset size
        # FAISS recommends at least 30 * nlist vectors for training
        ntotal = len(vectors_array)
        nlist = min(100, max(1, ntotal // 30))

        # Create IVF index for combined data
        quantizer = faiss.IndexFlatL2(vectors_array.shape[1])
        self.index = faiss.IndexIVFFlat(quantizer, vectors_array.shape[1], nlist)
        self.index.train(vectors_array)
        self.index.add(vectors_array)
        self.index.make_direct_map()
        faiss.write_index(self.index, f"./data/{self.args.base_data}/combined.index")
        
        # Save metadata JSON
        self.fm.save_json(metadata, f'./data/{self.args.base_data}/rag_metadata.json')
        
        return metadata

    def search_similar(self, query_features=None, query_signal=None, k=5, mode='query_signal',nprobe=10):
        
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
            original_indices = [int(self.index_mapping[idx]) for idx in indices[0]]
            
        elif mode == 'signal':
            self.signal_index.nprobe = nprobe
            query_signal = query_signal.reshape(1, -1)
            distances, indices = self.signal_index.search(query_signal, k)
            original_indices = [int(self.signal_mapping[idx]) for idx in indices[0]]
            
        else:  # combined mode
            self.index.nprobe = nprobe
            query_combined = np.hstack([query_features, query_signal])
            query_combined = query_combined.reshape(1, -1)
            distances, indices = self.index.search(query_combined, k)
            original_indices = [int(idx) for idx in indices[0]]


    
        # Prepare results using reconstructed vectors from index
        results = {}
        for i, (dist, idx) in enumerate(zip(distances[0], original_indices)):
            full_vector = self.index.reconstruct(int(idx))
            features = full_vector[:self.feature_dim]
            signal = full_vector[self.feature_dim:]
            
            result_dict = {
                'signal': signal,
                'feature': features,
                'report': self.reports[idx],
                'distance': float(dist),
                'file_path': self.file_paths[idx]
            }
            results[i] = result_dict
        
        return results
    
    def format_search(self, results):
        results = self.filter_results(results)
        # output = f"The following is the top {len(results)} retrieved ECGs and their corresponding features and diagnosis. Utilize this information to further enhance your response.\n\n"
        output = f"The following is the top {len(results)} retrieved ECGs and their corresponding diagnosis. Utilize this information to further enhance your response.\n\n"
        
        for idx, res in results.items():
            # Filter out entries where all feature values are zero
            if np.all(np.array(res['feature']) == 0):
                continue
            
            output += f"Retrieved ECG {idx+1}\n"
            output += "Diagnosis Information:\n"
            # output += f"{res['report']}\n\n"
            output += f"{res['report']}\n"
            # output += "Feature Information:\n"
            # Zip through feature names and feature values to format each line.
            # for feature_name, feature_value in zip(self.ecg_feature_list, res['feature']):
            #     output += f"{feature_name}: {str(float(feature_value))}\n"
            # output += "\n"
        return output
    
    def filter_results(self, results):
        filtered_results = {}
        count = 0
        for idx, res in results.items():
            # Check if more than x% of values are exactly zero or if the sum is too small
            feature_array = np.array(res['feature'])
            zero_percentage = np.sum(np.abs(feature_array) < 1e-3) / len(feature_array)
            total_magnitude = np.sum(np.abs(feature_array))
            
            # Filter out entries that are mostly zeros or have very low total magnitude
            if zero_percentage > 0.5 or total_magnitude < 0.5:
                continue
            
            filtered_results[count] = res
            count += 1
        return filtered_results

    def test_search(self):
        self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        npy_files = list(Path(self.preprocessed_dir).glob('*.npy'))
        random_idx = np.random.randint(0, len(npy_files))
        query_signal = self.fm.open_npy(npy_files[random_idx])['ecg']
        print('query_signal', query_signal.shape)
        # Flatten the signal to match the expected dimensions
        # query_signal = query_signal.flatten()
        start_time = time.time()
        results = self.search_similar(query_signal=query_signal, k=10, mode='signal')
        formatted_results = self.format_search(results)
        print(formatted_results)
        end_time = time.time()
        print(f"Search time: {end_time - start_time:.2f} seconds")