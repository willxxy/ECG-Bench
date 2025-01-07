import pytest
import numpy as np
import json
import pickle
import os
import tempfile
from pathlib import Path
from ecg_bench.utils.dir_file_utils import FileManager  # Updated import path

# Rest of your test fixtures
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def sample_json_file(temp_dir):
    data = {"test": "value", "number": 42}
    file_path = os.path.join(temp_dir, "test.json")
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def sample_npy_file(temp_dir):
    data = np.array([[1, 2], [3, 4]])
    file_path = os.path.join(temp_dir, "test.npy")
    np.save(file_path, data)
    return file_path

@pytest.fixture
def sample_pickle_file(temp_dir):
    vocab = {"word": 1, "test": 2}
    merges = {"a": "b", "c": "d"}
    file_path = os.path.join(temp_dir, "test.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump((vocab, merges), f)
    return file_path

@pytest.fixture
def signal_text_dirs(temp_dir):
    # Create signal and text directories with matching files
    signal_dir = os.path.join(temp_dir, "signals")
    text_dir = os.path.join(temp_dir, "texts")
    os.makedirs(signal_dir)
    os.makedirs(text_dir)
    
    # Create matching files
    for i in range(3):
        np.save(os.path.join(signal_dir, f"{i}_0.npy"), np.zeros(10))
        with open(os.path.join(text_dir, f"{i}_0.json"), 'w') as f:
            json.dump({"text": f"test_{i}"}, f)
    
    # Create some unmatched files
    np.save(os.path.join(signal_dir, "999_0.npy"), np.zeros(10))
    with open(os.path.join(text_dir, "888_0.json"), 'w') as f:
        json.dump({"text": "unmatched"}, f)
    
    return signal_dir, text_dir

@pytest.fixture
def sample_ecg_file(temp_dir):
    # Create a simple dummy ECG file - note that this is just a mock
    # In real testing, you'd need actual WFDB format files
    file_path = os.path.join(temp_dir, "ecg_test")
    return file_path

# Test cases
def test_open_json(sample_json_file):
    data = FileManager.open_json(sample_json_file)
    assert data["test"] == "value"
    assert data["number"] == 42

def test_open_npy(sample_npy_file):
    data = FileManager.open_npy(sample_npy_file)
    assert np.array_equal(data, np.array([[1, 2], [3, 4]]))

def test_load_vocab_and_merges(sample_pickle_file):
    vocab, merges = FileManager.load_vocab_and_merges(sample_pickle_file)
    assert vocab == {"word": 1, "test": 2}
    assert merges == {"a": "b", "c": "d"}

def test_ensure_directory_exists(temp_dir):
    new_dir = os.path.join(temp_dir, "new_directory")
    FileManager.ensure_directory_exists(new_dir)
    assert os.path.exists(new_dir)
    # Test idempotency
    FileManager.ensure_directory_exists(new_dir)
    assert os.path.exists(new_dir)

def test_align_signal_text_files(signal_text_dirs):
    signal_dir, text_dir = signal_text_dirs
    signal_files, text_files = FileManager.align_signal_text_files(signal_dir, text_dir)
    
    assert len(signal_files) == len(text_files) == 3
    for sig_file, txt_file in zip(signal_files, text_files):
        assert os.path.basename(sig_file).split('.')[0] == os.path.basename(txt_file).split('.')[0]

def test_sample_N_percent():
    items = list(range(100))
    sampled = FileManager.sample_N_percent(items, 0.1)
    assert len(sampled) == 10
    assert all(item in items for item in sampled)
    
    # Test minimum size
    small_items = [1, 2]
    sampled = FileManager.sample_N_percent(small_items, 0.1)
    assert len(sampled) == 1

def test_sample_N_percent_invalid():
    with pytest.raises(ValueError):
        FileManager.sample_N_percent([1, 2, 3], 1.5)

def test_sample_N_percent_from_lists():
    list1 = list(range(100))
    list2 = [str(x) for x in range(100)]
    
    # Test single list
    result1 = FileManager.sample_N_percent_from_lists(list1, N=0.1)
    assert len(result1) == 10
    
    # Test two lists
    result1, result2 = FileManager.sample_N_percent_from_lists(list1, list2, N=0.1)
    assert len(result1) == len(result2) == 10
    for x, y in zip(result1, result2):
        assert str(x) == y

def test_sample_N_percent_from_lists_unequal():
    with pytest.raises(ValueError):
        FileManager.sample_N_percent_from_lists([1, 2], [1, 2, 3])

def test_open_ecg():
    signal, fs = FileManager.open_ecg('./ecg_bench/data/ptb/records500/00000/00001_hr')
    
    assert isinstance(signal, np.ndarray)
    assert isinstance(fs, (int, float))
    assert fs > 0
    assert len(signal.shape) == 2   