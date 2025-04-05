#!/usr/bin/env python3
"""
Run all tests for ECG-Bench with nice output formatting.
This script allows running all tests or a specific subset of tests.
"""

import os
import sys
import subprocess
import argparse
from termcolor import colored
import time

TEST_MODULES = {
    "core": "test_core_components.py",
    "data": "test_data_loaders.py",
    "models": "test_models.py",
    "gpu": "test_gpu.py",
    "transformers": "test_transformers.py"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run ECG-Bench tests")
    parser.add_argument("--modules", type=str, nargs="+", choices=list(TEST_MODULES.keys()) + ["all"],
                      default=["all"], help="Test modules to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--xdist", "-x", action="store_true", help="Use pytest-xdist to parallelize tests")
    return parser.parse_args()

def run_tests(test_file, verbose=False, use_xdist=False):
    """Run a specific test file and return success status"""
    print(colored(f"Running {test_file}", "blue", attrs=["bold"]))
    start_time = time.time()
    
    cmd = ["pytest"]
    if verbose:
        cmd.append("-v")
    if use_xdist:
        cmd.append("-xvs")
    else:
        cmd.append("-s")
    cmd.append(test_file)
    
    try:
        # Run from the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(cmd, cwd=project_root)
        success = result.returncode == 0
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(colored(f"✓ {test_file} passed in {duration:.2f}s", "green"))
        else:
            print(colored(f"✗ {test_file} failed in {duration:.2f}s", "red"))
        
        return success
    except Exception as e:
        print(colored(f"Error running {test_file}: {e}", "red"))
        return False

def main():
    args = parse_args()
    
    print(colored("ECG-Bench Test Runner", "cyan", attrs=["bold"]))
    print(colored("=====================", "cyan"))
    
    # Determine which modules to run
    modules_to_run = list(TEST_MODULES.keys()) if "all" in args.modules else args.modules
    
    # Track results
    results = {}
    
    # Make sure we're in the right directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(colored(f"Running tests from: {project_root}", "cyan"))
    
    for module in modules_to_run:
        test_file = os.path.join("tests", TEST_MODULES[module])
        
        # Skip GPU tests if CUDA is not available
        if module == "gpu":
            import torch
            if not torch.cuda.is_available():
                print(colored("Skipping GPU tests - CUDA not available", "yellow"))
                continue
                
        # Skip transformers tests if directory doesn't exist
        if module == "transformers":
            transformers_dir = os.path.join(project_root, "transformers")
            if not os.path.isdir(transformers_dir):
                print(colored("Skipping transformers tests - directory not found", "yellow"))
                continue
        
        results[module] = run_tests(test_file, args.verbose, args.xdist)
    
    # Summary
    print("\n" + colored("Test Summary:", "cyan", attrs=["bold"]))
    print(colored("=============", "cyan"))
    
    all_passed = True
    for module, passed in results.items():
        status = colored("PASSED", "green") if passed else colored("FAILED", "red")
        print(f"{module}: {status}")
        all_passed = all_passed and passed
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main() 