import unittest
import numpy as np
from quantum_leap.tensor import Tensor

class TestAutograd(unittest.TestCase):

    def test_simple_add_backward(self):
        a = Tensor([2], requires_grad=True)
        b = Tensor([3], requires_grad=True)
        c = a + b
        c.backward()
        self.assertTrue(np.array_equal(a.grad, [1]))
        self.assertTrue(np.array_equal(b.grad, [1]))

    def test_simple_mul_backward(self):
        a = Tensor([2], requires_grad=True)
        b = Tensor([3], requires_grad=True)
        c = a * b
        c.backward()
        self.assertTrue(np.array_equal(a.grad, [3]))
        self.assertTrue(np.array_equal(b.grad, [2]))
        
    def test_complex_graph_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = Tensor(4.0, requires_grad=True)
        
        # e = (a*b) + c
        d = a * b
        e = d + c
        
        e.backward()
        
        # d_e/d_a = b = 3
        # d_e/d_b = a = 2
        # d_e/d_c = 1
        self.assertAlmostEqual(a.grad, 3.0)
        self.assertAlmostEqual(b.grad, 2.0)
        self.assertAlmostEqual(c.grad, 1.0)
        
if __name__ == '__main__':
    unittest.main()
