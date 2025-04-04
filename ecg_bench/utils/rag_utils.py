from ecg_bench.utils.preprocess_utils import ECGFeatureExtractor
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

class RAGECGDatabase:
    def __init__(self, args, fm):
        self.args = args
        self.fm = fm
        self.preprocessed_dir = f"./data/{self.args.base_data}/preprocessed_{self.args.seg_len}_{self.args.target_sf}"
        self.feature_extractor = ECGFeatureExtractor(self.args.target_sf)
        
        print('Loading RAG database...')
        if self.args.load_rag_db_idx is None and self.args.load_rag_db is None:
            print('No RAG database found. Creating new one...')
            self.metadata = self.create_and_save_db()
        else:
            print('Loading RAG database from file...')
            self.metadata = self.fm.open_json(self.args.load_rag_db)
            self.index = faiss.read_index(self.args.load_rag_db_idx)
        
        print('Metadata loaded.')
        self.reports = [item['report'] for item in self.metadata]
        self.file_paths = [item['file_path'] for item in self.metadata]
        
        print('RAG database loaded.')
        # Get dimensions from first vector in index
        first_vector = self.index.reconstruct(0)
        self.feature_dim = 288
        self.signal_dim = len(first_vector) - self.feature_dim
        
        print('features dim:', self.feature_dim)
        print('signals dim:', self.signal_dim)
        print('total samples:', len(self.reports))
        print('Index loaded.')

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
        self.index = faiss.IndexFlatL2(vectors_array.shape[1])
        self.index.add(vectors_array)
        faiss.write_index(self.index, f"./data/{self.args.base_data}/combined.index")
        
        # Save metadata JSON
        self.fm.save_json(metadata, f'./data/{self.args.base_data}/rag_metadata.json')
        
        return metadata

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
                
                self.feature_index = faiss.IndexFlatL2(self.feature_dim)
                self.feature_index.add(feature_vectors)
                self.index_mapping = np.arange(self.index.ntotal)
            
            query_features = query_features.reshape(1, self.feature_dim)
            distances, indices = self.feature_index.search(query_features, k)
            original_indices = [int(self.index_mapping[idx]) for idx in indices[0]]
            
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
            query_signal = query_signal.reshape(1, -1)  # Reshape to 2D array
            distances, indices = self.signal_index.search(query_signal, k)
            
            # Map back to original indices
            original_indices = [int(self.signal_mapping[idx]) for idx in indices[0]]
            
        else:  # combined mode
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
    
    def test_search(self):
        npy_files = list(Path(self.preprocessed_dir).glob('*.npy'))
        query_signal = self.fm.open_npy(npy_files[0])['ecg']
        print('query_signal', query_signal.shape)
        # Flatten the signal to match the expected dimensions
        # query_signal = query_signal.flatten()
        results = self.search_similar(query_signal=query_signal, k=5, mode='signal')
        print(results)