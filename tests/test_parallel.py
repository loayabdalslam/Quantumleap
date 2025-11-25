import unittest
import numpy as np
from quantum_leap.tensor import Tensor
from quantum_leap.parallel import parallel_matmul

class TestParallel(unittest.TestCase):

    def test_parallel_matmul_correctness(self):
        # Create two large matrices
        a = np.random.rand(200, 100)
        b = np.random.rand(100, 50)
        
        # Compute the result using both standard and parallel matmul
        expected = a @ b
        result = parallel_matmul(a, b)
        
        # Check that the results are close (allowing for floating point differences)
        self.assertTrue(np.allclose(result, expected))

    def test_tensor_parallel_integration(self):
        # Create two large tensors, which should trigger the parallel implementation
        t1 = Tensor(np.random.rand(200, 100))
        t2 = Tensor(np.random.rand(100, 50))
        
        # Compute the result using the tensor's matmul
        result_tensor = t1 @ t2
        
        # Compute the expected result using numpy
        expected = t1.data @ t2.data
        
        self.assertEqual(result_tensor.shape, expected.shape)
        self.assertTrue(np.allclose(result_tensor.data, expected))

if __name__ == '__main__':
    unittest.main()
