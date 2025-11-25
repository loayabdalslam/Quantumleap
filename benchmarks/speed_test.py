import time
import numpy as np
import sys
import os

# --- Add the project root to the Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- QuantumLeap ---
from quantum_leap.tensor import Tensor as qlTensor

# --- PyTorch ---
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- TensorFlow ---
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

def benchmark_matmul(framework, size=512):
    """
    Measures the time taken for matrix multiplication in a given framework.
    """
    if framework == 'quantum_leap':
        a = qlTensor(np.random.rand(size, size))
        b = qlTensor(np.random.rand(size, size))

        start_time = time.time()
        c = a @ b
        end_time = time.time()

    elif framework == 'torch' and HAS_TORCH:
        a = torch.rand(size, size)
        b = torch.rand(size, size)

        start_time = time.time()
        c = a @ b
        end_time = time.time()

    elif framework == 'tensorflow' and HAS_TENSORFLOW:
        a = tf.random.uniform((size, size))
        b = tf.random.uniform((size, size))

        start_time = time.time()
        c = a @ b
        end_time = time.time()

    else:
        return None # Framework not available

    return end_time - start_time

if __name__ == "__main__":
    print("--- Performance Benchmark: Matrix Multiplication ---")

    # QuantumLeap
    ql_time = benchmark_matmul('quantum_leap')
    print(f"QuantumLeap: {ql_time:.6f} seconds")

    # PyTorch
    if HAS_TORCH:
        torch_time = benchmark_matmul('torch')
        print(f"PyTorch:     {torch_time:.6f} seconds")
    else:
        print("PyTorch not found. Skipping benchmark.")

    # TensorFlow
    if HAS_TENSORFLOW:
        tf_time = benchmark_matmul('tensorflow')
        print(f"TensorFlow:  {tf_time:.6f} seconds")
    else:
        print("TensorFlow not found. Skipping benchmark.")

    # --- Comparison ---
    if HAS_TORCH and ql_time and torch_time:
        if ql_time < torch_time:
            print(f"\nQuantumLeap is {torch_time / ql_time:.2f}x faster than PyTorch!")
        else:
            print(f"\nPyTorch is {ql_time / torch_time:.2f}x faster than QuantumLeap.")

    if HAS_TENSORFLOW and ql_time and tf_time:
        if ql_time < tf_time:
            print(f"QuantumLeap is {tf_time / ql_time:.2f}x faster than TensorFlow!")
        else:
            print(f"TensorFlow is {ql_time / tf_time:.2f}x faster than QuantumLeap.")
