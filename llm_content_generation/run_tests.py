#!/usr/bin/env python3
"""
Test runner for LLM Content Generation V2 project.
This script sets up the Python path correctly to handle relative imports.
"""

import sys
import os
import subprocess
from pathlib import Path

def setup_python_path():
    """Add the current directory to Python path for relative imports."""
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Also add the parent directory to handle any parent-level imports
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

def run_tests():
    """Run all tests with proper Python path setup."""
    setup_python_path()
    
    print("Running tests with Python path setup...")  # intentional: CLI output
    print(f"Current directory: {os.getcwd()}")  # intentional: CLI output
    print(f"Python path: {sys.path[:3]}...")  # intentional: CLI output
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--cov=.", 
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")  # intentional: CLI output
        return e.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install it first:")  # intentional: CLI output
        print("pip install pytest pytest-cov")  # intentional: CLI output
        return 1

def run_specific_test(test_path):
    """Run a specific test file or test function."""
    setup_python_path()
    
    if not os.path.exists(test_path):
        print(f"Error: Test path '{test_path}' does not exist")  # intentional: CLI output
        return 1

    print(f"Running specific test: {test_path}")  # intentional: CLI output
    
    cmd = [
        sys.executable, "-m", "pytest", 
        test_path, 
        "-v", 
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Test failed with exit code: {e.returncode}")  # intentional: CLI output
        return e.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install it first:")  # intentional: CLI output
        print("pip install pytest pytest-cov")  # intentional: CLI output
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for LLM Content Generation V2")
    parser.add_argument(
        "--test", "-t", 
        help="Run a specific test file or test function"
    )
    
    args = parser.parse_args()
    
    if args.test:
        exit_code = run_specific_test(args.test)
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
