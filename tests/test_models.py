import pytest
import os
import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace

# Import ECG tokenizer if needed for model tests
try:
    from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


class SimpleECGEncoder(nn.Module):
    """A simple encoder for testing purposes"""
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, input_dim, seq_len)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        # Take the mean over the sequence dimension
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x


def test_simple_model():
    """Test a simple model to verify PyTorch works"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple model
    model = SimpleECGEncoder().to(device)
    print(f"✓ Created model: {model.__class__.__name__}")
    
    # Create a random input
    batch_size = 2
    input_dim = 12
    seq_len = 1000
    x = torch.randn(batch_size, input_dim, seq_len).to(device)
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (batch_size, 128), f"Expected output shape (2, 128), got {output.shape}"
    except Exception as e:
        pytest.fail(f"Model forward pass failed: {e}")


def test_model_save_load():
    """Test saving and loading a model"""
    # Create model
    model = SimpleECGEncoder()
    
    # Create a temporary directory
    os.makedirs("temp_model", exist_ok=True)
    model_path = "temp_model/test_model.pth"
    
    try:
        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Load model
        loaded_model = SimpleECGEncoder()
        loaded_model.load_state_dict(torch.load(model_path))
        print(f"✓ Model loaded from {model_path}")
        
        # Verify both models have the same parameters
        for (name1, p1), (name2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
            assert torch.allclose(p1, p2), f"Parameters don't match for {name1}"
        
        print("✓ Original and loaded models have the same parameters")
        
        # Clean up
        import shutil
        shutil.rmtree("temp_model", ignore_errors=True)
    except Exception as e:
        pytest.fail(f"Model save/load test failed: {e}")


def test_optimizer_and_training_loop():
    """Test basic optimizer and training loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SimpleECGEncoder().to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Mock data
    batch_size = 4
    input_dim = 12
    seq_len = 1000
    x = torch.randn(batch_size, input_dim, seq_len).to(device)
    target = torch.randn(batch_size, 128).to(device)  # Random targets
    
    try:
        # Training loop
        initial_loss = None
        final_loss = None
        
        for epoch in range(5):  # Just a few epochs for testing
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            
            if epoch == 0:
                initial_loss = loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            print(f"Epoch {epoch+1}/5, Loss: {loss.item():.6f}")
        
        print(f"Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")
        assert final_loss < initial_loss, "Training didn't reduce the loss"
        print("✓ Training loop successfully reduced loss")
    except Exception as e:
        pytest.fail(f"Training loop test failed: {e}")


if __name__ == "__main__":
    print("Running model tests...")
    test_simple_model()
    test_model_save_load()
    test_optimizer_and_training_loop()
    print("All model tests completed!") 