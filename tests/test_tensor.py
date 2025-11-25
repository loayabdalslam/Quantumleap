import unittest
import numpy as np
import sys
import os

# --- Add the project root to the Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_leap.tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_add(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        self.assertTrue(np.array_equal(c.data, np.array([5, 7, 9])))

    def test_mul(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a * b
        self.assertTrue(np.array_equal(c.data, np.array([4, 10, 18])))

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a @ b
        self.assertTrue(np.array_equal(c.data, np.array([[19, 22], [43, 50]])))

if __name__ == '__main__':
    unittest.main()
