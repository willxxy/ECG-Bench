from ecg_bench.utils.preprocess_utils import ECGFeatureExtractor
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

class RAGECGDatabse:
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.feature_extractor = ECGFeatureExtractor(self.args.target_sf)
        if self.args.load_rag_db == None:
            self.all_data = self.get_features()
        else:
            self.all_data = self.fm.open_json(self.args.load_rag_db)
        
        self.features = np.stack([item['features'] for item in self.all_data])
        self.signals = np.stack([item['signal'] for item in self.all_data])
        self.reports = [item['report'] for item in self.all_data]
        self.file_path = [item['file_path'] for item in self.all_data]
        self.feature_dim = self.features.shape[1]
        self.signal_dim = self.signals.shape[1]
        print('features', self.features.shape)
        print('signals', self.signals.shape)
        print('reports', len(self.reports))
        print('file_path', len(self.file_path))
        print('feature_dim', self.feature_dim)
        print('signal_dim', self.signal_dim)
        if self.args.load_rag_db_idx == None:
            self.create_and_save_db()
        else:
            self.index = faiss.read_index(self.args.load_rag_db_idx)
        
    def create_and_save_db(self):
        combined_data = np.hstack([self.features, self.signals])
        self.index = faiss.IndexFlatL2(combined_data.shape[1])
        self.index.add(combined_data)
        faiss.write_index(self.index, f"./data/{self.args.base_data}/combined.index")
        print(f"Saved combined index to ./data/{self.args.base_data}/combined.index")
        
    def get_features(self):
        all_data = []
        npy_files = list(Path(self.preprocessed_dir).glob('*.npy'))
        if self.args.dev:
            npy_files = npy_files[:1000]
        if self.args.toy:
            npy_files = npy_files[:400000]
        print(f"Found {len(npy_files)} .npy files in {self.preprocessed_dir}")
        if len(npy_files) == 0:
            raise ValueError(f"No .npy files found in directory: {self.preprocessed_dir}")
        
        for file_path in tqdm(npy_files, desc="Extracting features"):
            try:
                data = self.fm.open_npy(file_path)
                ecg = data['ecg']
                report = data['report']
                features = self.feature_extractor.extract_features(ecg)
                all_data.append({
                                'features': features.flatten().tolist(),
                                'signal': ecg.flatten().tolist(),
                                'report': report,
                                'file_path': str(file_path)
                                })
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        self.fm.save_json(all_data, f'./data/{self.args.base_data}/all_data_rag_db.json')
        return all_data
    
    def search_similar(self, query_features=None, query_signal=None, k=5, mode='query_signal'):
        """
        Search for similar ECGs using different modes
        
        Args:
            query_features: feature vector for similarity search
            query_signal: signal vector for similarity search
            k: number of results to return
            mode: 'feature', 'signal', or 'combined'
        """
        if mode not in ['feature', 'signal', 'combined']:
            raise ValueError("Mode must be 'feature', 'signal', or 'combined'")
            
        if mode == 'feature' and query_features is None:
            raise ValueError("Feature mode requires query_features")
        if mode == 'signal' and query_signal is None:
            raise ValueError("Signal mode requires query_signal")
        if mode == 'combined' and (query_features is None or query_signal is None):
            raise ValueError("Combined mode requires both query_features and query_signal")

        if mode == 'feature':
            if not hasattr(self, 'feature_index'):
                # Extract only feature part from all vectors
                feature_vectors = np.zeros((self.index.ntotal, self.feature_dim), dtype=np.float32)
                for i in range(self.index.ntotal):
                    full_vector = self.index.reconstruct(i)
                    feature_vectors[i] = full_vector[:self.feature_dim]
                
                # Create and store feature-only index
                self.feature_index = faiss.IndexFlatL2(self.feature_dim)
                self.feature_index.add(feature_vectors)
                
                # Store mapping from feature index to full index
                self.index_mapping = np.arange(self.index.ntotal)
            
            # Search using feature index
            query_features = query_features.reshape(1, self.feature_dim)
            distances, indices = self.feature_index.search(query_features, k)
            
            # Map back to original indices
            original_indices = [self.index_mapping[idx] for idx in indices[0]]
            
        elif mode == 'signal':
            if not hasattr(self, 'signal_index'):
                # Extract only signal part from all vectors
                signal_vectors = np.zeros((self.index.ntotal, self.signal_dim), dtype=np.float32)
                for i in range(self.index.ntotal):
                    full_vector = self.index.reconstruct(i)
                    signal_vectors[i] = full_vector[self.feature_dim:]
                
                # Create and store signal-only index
                self.signal_index = faiss.IndexFlatL2(self.signal_dim)
                self.signal_index.add(signal_vectors)
                
                # Store mapping from signal index to full index
                self.signal_mapping = np.arange(self.index.ntotal)
            
            # Search using signal index
            query_signal = query_signal.reshape(1, self.signal_dim)
            distances, indices = self.signal_index.search(query_signal, k)
            
            # Map back to original indices
            original_indices = [self.signal_mapping[idx] for idx in indices[0]]
            
        else:  # combined mode
            query_combined = np.hstack([query_features, query_signal])
            query_combined = query_combined.reshape(1, -1)
            distances, indices = self.index.search(query_combined, k)
            original_indices = indices[0]

        # Prepare results
        results = {}
        for i, (dist, idx) in enumerate(zip(distances[0], original_indices)):
            result_dict = {
                f'signal': self.signals[idx],
                f'feature': self.features[idx],
                f'report': self.reports[idx],
                f'distance': float(dist),
                f'file_path': self.file_path[idx]
            }
            results[i] = result_dict
        
        return results
    
    def test_search(self):
        query_signal = self.signals[0]
        print('query_signal', query_signal.shape)
        results = self.search_similar(query_signal=query_signal, k=5, mode='signal')
        print(results)
        print('-----' * 100)
        
        npy_files = list(Path(self.preprocessed_dir).glob('*.npy'))
        query_signal2 = self.fm.open_npy(npy_files[0])['ecg']
        results2 = self.search_similar(query_signal=query_signal2, k=5, mode='signal')
        print(results2)
        print('-----' * 100)
        for key in results.keys():
            print(results[key]['file_path'] == results2[key]['file_path'])