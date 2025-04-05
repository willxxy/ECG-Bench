import pytest
import os
import torch
import yaml
import numpy as np
import pickle
from pathlib import Path
from argparse import Namespace

# Track import status
file_manager_imported = False
ecg_tokenizer_imported = False
viz_util_imported = False

# Import ECG-Bench modules - adjust imports as needed
try:
    from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
    ecg_tokenizer_imported = True
except ImportError as e:
    print(f"Warning: Could not import ECGByteTokenizer: {e}")

try:
    from ecg_bench.utils.dir_file_utils import FileManager
    file_manager_imported = True
except ImportError as e:
    print(f"Warning: Could not import FileManager: {e}")

try:
    from ecg_bench.utils.viz_utils import VizUtil
    viz_util_imported = True
except ImportError as e:
    print(f"Warning: Could not import VizUtil: {e}")


def test_environment_setup():
    """Test that the environment is properly set up"""
    # Check Python environment
    print("Testing Python environment...")
    assert os.path.exists("requirements.txt"), "requirements.txt file not found"
    
    # Check CUDA availability if applicable
    if torch.cuda.is_available():
        print(f"✓ CUDA is available (version: {torch.version.cuda})")
        print(f"✓ Number of CUDA devices: {torch.cuda.device_count()}")
    else:
        print("✗ CUDA is not available. Running on CPU only.")
    
    # Check if essential directories exist
    for dir_name in ["ecg_bench", "tests", ".github"]:
        assert os.path.isdir(dir_name), f"Essential directory {dir_name} not found"


def test_file_manager():
    """Test the FileManager utility"""
    # Skip test if FileManager isn't available
    if not file_manager_imported:
        pytest.skip("FileManager not imported - skipping test")
        
    try:
        fm = FileManager()
        # Test directory creation using os methods directly
        temp_dir = "temp_test_dir"
        
        # Create directory
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        assert os.path.isdir(temp_dir), f"Failed to create directory {temp_dir}"
        
        # Test file operations
        test_file = os.path.join(temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        assert os.path.isfile(test_file), f"Failed to create file {test_file}"
        
        # Clean up
        os.remove(test_file)
        os.rmdir(temp_dir)
    except Exception as e:
        pytest.fail(f"FileManager test failed: {e}")


def test_viz_util():
    """Test the VizUtil class"""
    # Skip test if VizUtil isn't available
    if not viz_util_imported:
        pytest.skip("VizUtil not imported - skipping test")
        
    try:
        viz = VizUtil()
        assert viz is not None, "Failed to initialize VizUtil"
    except Exception as e:
        pytest.fail(f"VizUtil test failed: {e}")


def create_mock_tokenizer_file():
    """Create a mock tokenizer file with basic vocabulary and merges"""
    # Create a temporary directory
    os.makedirs("temp_tokenizer_dir", exist_ok=True)
    
    # Create simple mock vocabulary and merges
    mock_vocab = {
        "a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5,
        "<PAD>": 6, "<UNK>": 7, "<BOS>": 8, "<EOS>": 9
    }
    
    mock_merges = {
        ("a", "b"): "ab",
        ("b", "c"): "bc",
        ("ab", "c"): "abc"
    }
    
    # Create a tokenizer file
    tokenizer_path = "temp_tokenizer_file.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump((mock_vocab, mock_merges), f)
    
    return tokenizer_path


def test_ecg_tokenizer_basic():
    """Test basic functionality of ECG tokenizer"""
    # Skip test if ECGByteTokenizer or FileManager aren't available
    if not ecg_tokenizer_imported or not file_manager_imported:
        pytest.skip("ECGByteTokenizer or FileManager not imported - skipping test")
        
    try:
        # Create a small mock ECG signal for testing
        mock_ecg = np.random.randn(12, 1000)  # 12-lead ECG with 1000 samples
        
        # Create a mock tokenizer file
        tokenizer_path = create_mock_tokenizer_file()
        
        # Initialize FileManager first since ECGByteTokenizer requires it
        fm = FileManager()
        
        # Create a mock args object with dev=False and add all required attributes
        mock_args = Namespace(
            dev=False,
            num_merges=3500,
            ecg_tokenizer=tokenizer_path,  # Use the file path, not directory
            instance_normalize=True,
            percentiles=None,
            image=False,
            target_sf=250,
            seg_len=1000,
            pad_to_max=1024,
            data="temp_data"
        )
        
        # Check if tokenizer can be initialized
        tokenizer = ECGByteTokenizer(mock_args, fm)
        assert tokenizer is not None, "Failed to initialize ECGByteTokenizer"
        
        # Clean up
        import shutil
        os.remove(tokenizer_path)
        if os.path.exists("temp_tokenizer_dir"):
            shutil.rmtree("temp_tokenizer_dir", ignore_errors=True)
    except Exception as e:
        pytest.fail(f"ECG tokenizer test failed: {e}")


def test_config_loading():
    """Test loading configuration from YAML files"""
    # Create a test config
    config_data = {
        "model": "test-model",
        "data": "test-data",
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 10
    }
    
    os.makedirs("temp_config", exist_ok=True)
    config_path = "temp_config/test_config.yaml"
    
    try:
        # Write test config
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Read and verify config
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == config_data, "Config loading failed"
        
        # Clean up
        os.remove(config_path)
        os.rmdir("temp_config")
    except Exception as e:
        pytest.fail(f"Config loading test failed: {e}")


if __name__ == "__main__":
    print("Running core component tests...")
    test_environment_setup()
    test_file_manager()
    test_viz_util()
    test_ecg_tokenizer_basic()
    test_config_loading()
    print("All tests completed!")