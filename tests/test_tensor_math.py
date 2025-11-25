import unittest
import numpy as np
from quantum_leap.tensor import Tensor, zeros, ones, rand

class TestTensorMath(unittest.TestCase):

    def test_addition(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 + t2
        self.assertTrue(np.array_equal(result.data, [5, 7, 9]))

    def test_multiplication(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 * t2
        self.assertTrue(np.array_equal(result.data, [4, 10, 18]))

    def test_matrix_multiplication(self):
        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([[5, 6], [7, 8]])
        result = t1 @ t2
        expected = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result.data, expected))

if __name__ == '__main__':
    unittest.main()
