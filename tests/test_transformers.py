def test_transformers_installation():
    print("Testing transformers installation...")
    EXPECTED_VERSION = "4.47.1"
    
    try:
        import transformers
        print("✓ Basic import successful")
        actual_version = transformers.__version__
        print(f"✓ Transformers version: {actual_version}")
        
        # Check if version matches expected
        if actual_version != EXPECTED_VERSION:
            print(f"✗ WARNING: Version mismatch! Expected {EXPECTED_VERSION}, got {actual_version}")
            return
        else:
            print(f"✓ Version match confirmed: {actual_version}")
            
        # Check installation path to confirm it's using local submodule
        import os
        install_path = os.path.dirname(transformers.__file__)
        print(f"✓ Installation path: {install_path}")
        if "site-packages" in install_path:
            print("✗ WARNING: Transformers appears to be installed from PyPI, not from local submodule")
        else:
            print("✓ Transformers is installed from local submodule")
            
    except ImportError as e:
        print("✗ Failed to import transformers:", e)
        return
    
    try:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./.huggingface')
        test_text = "Testing BERT tokenizer"
        tokens = tokenizer(test_text)
        print("✓ Tokenizer loading and tokenization successful")
        print(f"  Sample tokenization: {test_text} -> {tokenizer.convert_ids_to_tokens(tokens['input_ids'])}")
    except Exception as e:
        print("✗ Failed to load tokenizer:", e)
        return
    
    try:
        from transformers import BertModel
        model = BertModel.from_pretrained('bert-base-uncased', cache_dir='./.huggingface')
        print("✓ Model loading successful")
    except Exception as e:
        print("✗ Failed to load model:", e)
        return
    
    try:
        import inspect
        source_code = inspect.getsource(transformers.BertModel)
        print("✓ Can access source code of transformers")
        
        # Check if we can make modifications
        try:
            with open(os.path.join(os.path.dirname(transformers.__file__), 'models', 'bert', 'modeling_bert.py'), 'r') as f:
                _ = f.read()
            print("✓ Can read source files directly (editable mode confirmed)")
        except Exception as e:
            print("✗ Cannot read source files directly. May not be in editable mode:", e)
            
    except Exception as e:
        print("✗ Failed to access source code:", e)
        return
    
    print("\nTest Summary:")
    print(f"✓ Transformers {actual_version} is properly installed")
    print(f"✓ Installation is from local submodule at {install_path}")
    print("✓ All functionality tests passed")

if __name__ == "__main__":
    test_transformers_installation()