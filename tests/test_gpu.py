import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import os

def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    return dist.is_initialized()

def test_distributed_tensor(rank, world_size):
    """Test distributed tensor operations"""
    try:
        # Create tensor on each process
        tensor = torch.ones(2, 2) * rank
        print(f"Process {rank}: Initial tensor =\n{tensor}")

        # All-reduce operation
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(i * torch.ones(2, 2) for i in range(world_size))
        is_correct = torch.allclose(tensor, expected_sum)
        print(f"Process {rank}: After all_reduce =\n{tensor}")
        print(f"Process {rank}: Result is correct: {is_correct}")
        
        return is_correct
    except Exception as e:
        print(f"Process {rank}: Error in distributed tensor test: {e}")
        return False

def test_flash_attention():
    """Test Flash Attention availability and functionality"""
    print("\nFlash Attention Test")
    print("====================")
    
    try:
        # Check if Flash Attention is available
        try:
            from flash_attn import flash_attn_func
            print("✓ Flash Attention package is installed")
            
            # Check Flash Attention version
            try:
                import flash_attn
                print(f"✓ Flash Attention version: {flash_attn.__version__}")
            except Exception:
                print("! Could not determine Flash Attention version")
            
            # Check if CUDA is available (required for Flash Attention)
            if not torch.cuda.is_available():
                print("✗ CUDA is not available, Flash Attention requires CUDA")
                return False
            else:
                print(f"✓ CUDA is available (version: {torch.version.cuda})")
                
                # Test Flash Attention with a small example
                try:
                    # Flash Attention expects tensors in [batch_size, seq_len, num_heads, head_dim]
                    # or [batch_size, seq_len, hidden_dim] format
                    batch_size = 2
                    seq_len = 16
                    num_heads = 8
                    head_dim = 64
                    
                    # Create tensors in the correct format for flash_attn_func
                    # [batch_size, seq_len, num_heads, head_dim]
                    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                    
                    print(f"Query tensor shape: {q.shape}")
                    
                    # Run flash attention
                    out = flash_attn_func(q, k, v)
                    print("✓ Flash Attention computation successful")
                    print(f"  Output shape: {out.shape}")
                    
                    # Test with longer sequence length
                    long_seq_len = 128
                    print(f"\nTesting with longer sequence length: {long_seq_len}")
                    q_long = torch.randn(batch_size, long_seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                    k_long = torch.randn(batch_size, long_seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                    v_long = torch.randn(batch_size, long_seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                    
                    # Measure time for Flash Attention
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    out_long = flash_attn_func(q_long, k_long, v_long)
                    end_time.record()
                    
                    torch.cuda.synchronize()
                    flash_attn_time = start_time.elapsed_time(end_time)
                    print(f"✓ Flash Attention with long sequence successful")
                    print(f"  Time taken: {flash_attn_time:.2f} ms")
                    
                    print("\n✓ Flash Attention test completed successfully")
                    return True
                        
                except Exception as e:
                    print(f"✗ Failed to run Flash Attention computation: {e}")
                    
                    # Try alternative format if the first attempt failed
                    try:
                        print("\nTrying alternative tensor format...")
                        # Some versions of flash_attn may expect different tensor formats
                        # Try the format [batch_size, seq_len, hidden_dim]
                        batch_size = 2
                        seq_len = 16
                        hidden_dim = 512
                        
                        q = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
                        k = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
                        v = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
                        
                        print(f"Alternative query tensor shape: {q.shape}")
                        
                        # Try with causal=True option which some versions require
                        out = flash_attn_func(q, k, v, causal=True)
                        print("✓ Flash Attention computation successful with alternative format")
                        print(f"  Output shape: {out.shape}")
                        return True
                    except Exception as e2:
                        print(f"✗ Failed with alternative format as well: {e2}")
                        
                        # Try one more approach - using the FlashAttention class directly
                        try:
                            print("\nTrying FlashAttention class directly...")
                            from flash_attn.flash_attention import FlashAttention
                            
                            flash_attn = FlashAttention()
                            batch_size = 2
                            seq_len = 16
                            num_heads = 8
                            head_dim = 64
                            
                            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
                            
                            out = flash_attn(q, k, v)
                            print("✓ FlashAttention class computation successful")
                            return True
                        except Exception as e3:
                            print(f"✗ Failed with FlashAttention class as well: {e3}")
                            return False
        except ImportError:
            print("✗ Flash Attention is not installed")
            return False
    except Exception as e:
        print(f"✗ Error testing Flash Attention: {e}")
        return False

def check_pytorch_installation(rank=None, world_size=None):
    """Check PyTorch installation including distributed capabilities"""
    print("PyTorch Installation Check")
    print("==========================")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print("\nCUDA Information:")
        print("-----------------")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device ID: {current_device}")
        device = torch.device(f'cuda:{current_device}')
        print(f"Current CUDA device: {device}")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA device name: {device_name}")
        
        print("\nTensor Operations:")
        print("------------------")
        try:
            x = torch.tensor(1).to(device)
            print(f"Moved tensor: {x}")
            y = torch.tensor(1, device=device)
            print(f"Created tensor: {y}")
            if x.device.type == 'cuda' and y.device.type == 'cuda':
                print("\nSuccess: PyTorch is correctly installed with CUDA support!")
            else:
                print("\nWarning: Tensors were not properly moved to CUDA devices.")
        except Exception as e:
            print(f"\nError during tensor operations: {e}")
    else:
        print("\nNote: CUDA is not available. PyTorch will run on CPU only.")

    # Distributed Training Test
    if rank is not None:
        print(f"\nDistributed Training Test (Process {rank}):")
        print("----------------------------------------")
        try:
            is_initialized = setup_distributed(rank, world_size)
            print(f"Process {rank}: Distributed setup successful: {is_initialized}")
            
            if is_initialized:
                test_result = test_distributed_tensor(rank, world_size)
                if test_result:
                    print(f"Process {rank}: Distributed operations test passed!")
                else:
                    print(f"Process {rank}: Distributed operations test failed!")
                
                # Cleanup
                dist.destroy_process_group()
        except Exception as e:
            print(f"Process {rank}: Error in distributed setup: {e}")

    print("\nSystem Information:")
    print("-------------------")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {sys.platform}")

def run_distributed_test(world_size=2):
    """Launch distributed test with multiple processes"""
    mp.spawn(check_pytorch_installation,
            args=(world_size,),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    # Run regular installation check
    print("Running single-process installation check...")
    check_pytorch_installation()
    
    print("\n\nRunning distributed training check...")
    run_distributed_test(world_size=2)
    
    # Run Flash Attention test
    print("\n\nRunning Flash Attention test...")
    flash_attn_result = test_flash_attention()
    if flash_attn_result:
        print("Flash Attention test passed!")
    else:
        print("Flash Attention test failed or not available.")
    
    print('\n\nAll tests completed.')