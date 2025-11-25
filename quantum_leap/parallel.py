import numpy as np
import multiprocessing

# --- Constants ---
# Use parallel matmul for matrices where the product of dimensions exceeds this threshold
PARALLEL_THRESHOLD = 10000

def _matmul_worker(args):
    """Helper function for the multiprocessing pool."""
    a_chunk, b = args
    return a_chunk @ b

def parallel_matmul(a, b):
    """
    Performs matrix multiplication in parallel using multiple CPU cores.
    Splits matrix 'a' into chunks and processes them concurrently.
    """
    # Use all available CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Split the first matrix 'a' into chunks along its rows
    chunk_size = a.shape[0] // num_cores
    if chunk_size == 0: # Handle cases with fewer rows than cores
        chunk_size = 1
        
    chunks = [a[i:i + chunk_size] for i in range(0, a.shape[0], chunk_size)]
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Map the worker function to the chunks
        results = pool.map(_matmul_worker, [(chunk, b) for chunk in chunks])
    
    # Concatenate the results
    return np.vstack(results)
