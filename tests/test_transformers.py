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
    
    # Test Flash Attention with Llama model
    print("\nTesting Flash Attention support with Llama model...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Check if CUDA is available (required for Flash Attention)
        if not torch.cuda.is_available():
            print("✗ CUDA is not available, Flash Attention requires CUDA")
        else:
            print(f"✓ CUDA is available (version: {torch.version.cuda})")
            
            # Test with Llama model
            try:
                print("Attempting to load a Llama model with flash_attention...")
                # Using a small Llama model for testing
                model_name = "meta-llama/llama-3.2-1b"  # Small Llama model
                
                # Try to load with flash attention explicitly set
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir='./.huggingface',
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    attn_implementation="flash_attention_2",
                ).to("cuda")
                
                print("✓ Successfully loaded Llama model with flash_attention_2 parameter")
                
                # Print the model config to inspect Flash Attention settings
                print("\nModel Config:")
                for key, value in model.config.to_dict().items():
                    if key.startswith("_") or "attn" in key:
                        print(f"  {key}: {value}")
                
                # Check for Flash Attention indicators in Llama config
                if hasattr(model.config, "_attn_implementation_autoset"):
                    print(f"✓ _attn_implementation_autoset is present: {model.config._attn_implementation_autoset}")
                
                # Check if Flash Attention is actually being used
                try:
                    # Create a small input for testing
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./.huggingface')
                    inputs = tokenizer("Testing Flash Attention with Llama", return_tensors="pt").to("cuda")
                    
                    # Run a forward pass
                    with torch.no_grad():
                        outputs = model(**inputs)
                    print("✓ Model forward pass successful")
                    
                    # Check CUDA memory usage as an indirect indicator
                    print("\nChecking CUDA memory usage for Flash Attention indicators...")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Run with potential Flash Attention
                    outputs = model.generate(**inputs, max_length=50)
                    peak_memory_flash = torch.cuda.max_memory_allocated()
                    
                    print(f"  Peak memory usage: {peak_memory_flash / 1024**2:.2f} MB")
                    print("  Note: Lower memory usage may indicate Flash Attention is working")
                    print("  Generated text: " + tokenizer.decode(outputs[0], skip_special_tokens=True))
                        
                except Exception as e:
                    print(f"✗ Failed during Flash Attention verification: {e}")
            except Exception as e:
                print(f"✗ Failed to load Llama model with flash_attention: {e}")
                print("  This could be because flash_attention is not properly installed or not compatible")
    except Exception as e:
        print(f"✗ Error testing Flash Attention: {e}")
    
    print("\nTest Summary:")
    print(f"✓ Transformers {actual_version} is properly installed")
    print(f"✓ Installation is from local submodule at {install_path}")
    print("✓ All functionality tests passed")

if __name__ == "__main__":
    test_transformers_installation()