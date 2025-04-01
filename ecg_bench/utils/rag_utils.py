import numpy as np
import os
from pathlib import Path
import faiss
from tqdm import tqdm

def read_npy_files(directory):
    all_data = []
    npy_files = list(Path(directory).glob('*_0.npy'))
    print(f"Found {len(npy_files)} .npy files in {directory}")
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in directory: {directory}")
    #try
    for file_path in tqdm(npy_files, desc="Reading NPY files"):
        try:
            # Load the .npy file
            data = np.load(file_path, allow_pickle=True).item()
            
            # Get ECG signal data
            ecg = data['ecg'].flatten()
            report = data['report']
            # Extract features in order
            feature_list = []
            for feature_dict in data['features']:
                # Extract features in a specific order
                ordered_features = [
                    feature_dict['mean'],
                    feature_dict['std'],
                    feature_dict['max'],
                    feature_dict['min'],
                    feature_dict['median'],
                    feature_dict['25th_per'],
                    feature_dict['75th_per'],
                    feature_dict['total_power'],
                    feature_dict['peak_freq_power'],
                    feature_dict['dominant_freq'],
                    feature_dict['spectral_centroid'],
                    feature_dict['heart_rate'],
                    feature_dict['heart_rate_var'],
                    feature_dict['QRS'],
                    feature_dict['T_wave_amp'],
                    feature_dict['ST_sd'],
                    feature_dict['mean_abs_wav0'],
                    feature_dict['mean_abs_wav1'],
                    feature_dict['mean_abs_wav2'],
                    feature_dict['mean_abs_wav3'],
                    feature_dict['mean_abs_wav4'],
                    feature_dict['mean_abs_wav5'],
                    feature_dict['avg_abs_dif'],
                    feature_dict['RMS_success_dif']
                ]
                feature_list.extend(ordered_features)
            
            # Store features and ECG signal separately
            features = np.array(feature_list, dtype=np.float32)
            all_data.append({
                'features': features.flatten(),
                'signal': ecg,
                'report': report,
                'file_path': str(file_path)
            })

            
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    return all_data

class ECGDatabase:
    def __init__(self, data_list=None):
        if data_list is not None:
            self.data_list = data_list
            self.features = np.stack([item['features'] for item in data_list])
            self.signals = np.stack([item['signal'] for item in data_list])
            self.reports = [item['report'] for item in data_list]
            self.file_path = [item['file_path'] for item in data_list]
            # Store dimensions for later use
            self.feature_dim = self.features.shape[1]
            self.signal_dim = self.signals.shape[1]
            
            # Create combined data (features + signals)
            self.combined_data = np.hstack([self.features, self.signals])
            
            # Create single FAISS index for combined data
            self.index = faiss.IndexFlatL2(self.combined_data.shape[1])
            self.index.add(self.combined_data)

    def save_database(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "combined.index"))
        # Save combined data
        np.save(os.path.join(save_dir, "ecg_data.npy"), {
            'features': self.features,
            'signals': self.signals,
            'reports': self.reports,
            'file_path': self.file_path
        })
    
    @classmethod
    def load_database(cls, save_dir):
        db = cls(data_list=None)
        # Load FAISS index
        db.index = faiss.read_index(os.path.join(save_dir, "combined.index"))
        # Load combined data
        data = np.load(os.path.join(save_dir, "ecg_data.npy"), allow_pickle=True).item()
        db.features = data['features']
        db.signals = data['signals']
        db.reports = data['reports']
        db.file_path = data['file_path']
        db.feature_dim = db.features.shape[1]
        db.signal_dim = db.signals.shape[1]
        return db

    def search_similar(self, query_features=None, query_signal=None, k=5, mode='feature'):
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
                f'signal_{i}': self.signals[idx],
                f'feature_{i}': self.features[idx],
                f'report_{i}': self.reports[idx],
                f'distance_{i}': float(dist),
                f'file_path_{i}': self.file_path[idx]
            }
            results[i] = result_dict
        
        return results

def main():
    # 1. First time: Create and save database
    def create_and_save_db():
        directory = "data/mimic/preprocessed_1250_250"
        print("Creating database...")
        data_list = read_npy_files(directory)
        db = ECGDatabase(data_list)
        print("Saving database...")
        db.save_database("./data/ecg_database")
        return db

    # 2. Later: Load existing database and search
    def load_and_search():
        db = ECGDatabase.load_database("./data/ecg_database")
        
        # Create queries (example: using first ECG as query)
        query_features = db.features[0]
        query_signal = db.signals[0]
        
        # Search using different modes
        print("\nFeature-based Search:")
        feature_results = db.search_similar(query_features=query_features, mode='feature', k=5)
        print(feature_results)
        # for i in range(len(feature_results)):
        #     print(f"\nMatch {i+1}:")
        #     print(f"Distance: {feature_results[i][f'distance_{i}']:.4f}")
        #     print(f"Report: {feature_results[i][f'report_{i}']}")
        #     print(f"File Path: {feature_results[i][f'file_path_{i}']}")
        #     print("-" * 80)

        # print("\nSignal-based Search:")
        # signal_results = db.search_similar(query_signal=query_signal, mode='signal', k=5)
        # for i in range(len(signal_results)):
        #     print(f"\nMatch {i+1}:")
        #     print(f"Distance: {signal_results[i][f'distance_{i}']:.4f}")
        #     print(f"Report: {signal_results[i][f'report_{i}']}")
        #     print(f"File Path: {signal_results[i][f'file_path_{i}']}")
        #     print("-" * 80)

        # print("\nCombined Search:")
        # combined_results = db.search_similar(
        #     query_features=query_features,
        #     query_signal=query_signal,
        #     mode='combined',
        #     k=5
        # )

        # for i in range(len(combined_results)):
        #     print(f"\nMatch {i+1}:")
        #     print(f"Distance: {combined_results[i][f'distance_{i}']:.4f}")
        #     print(f"Report: {combined_results[i][f'report_{i}']}")
        #     print(f"File Path: {combined_results[i][f'file_path_{i}']}")
        #     print("-" * 80)

    # Uncomment the one you want to use:
    # create_and_save_db()  # First time
    load_and_search()     # Later uses

if __name__ == "__main__":
    main()