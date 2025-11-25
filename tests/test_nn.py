import unittest
import numpy as np
from quantum_leap.tensor import Tensor
from quantum_leap.nn import Module, Linear, ReLU

class TestNN(unittest.TestCase):

    def test_linear_layer_forward(self):
        layer = Linear(in_features=10, out_features=5)
        input_tensor = Tensor(np.random.rand(1, 10))
        output = layer(input_tensor)
        self.assertEqual(output.shape, (1, 5))

    def test_relu_forward(self):
        relu = ReLU()
        input_tensor = Tensor([-1, 0, 1, 2])
        output = relu(input_tensor)
        self.assertTrue(np.array_equal(output.data, [0, 0, 1, 2]))

    def test_model_parameters(self):
        class SimpleModel(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(10, 5)
                self.layer2 = Linear(5, 1)

        model = SimpleModel()
        params = list(model.parameters())
        # 2 weights + 2 biases = 4 tensors
        self.assertEqual(len(params), 4)

if __name__ == '__main__':
    unittest.main()
