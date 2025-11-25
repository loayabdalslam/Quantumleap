from .tensor import Tensor, rand

class Module:
    """
    Base class for all neural network modules.
    Your models should also subclass this class.
    """
    def __init__(self):
        self._parameters = {}

    def parameters(self):
        """
        Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        """
        for name, param in self._parameters.items():
            yield param
            
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                for param in module.parameters():
                    yield param

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        super().__setattr__(name, value)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        self.weight = rand(out_features, in_features, requires_grad=True)
        if bias:
            self.bias = rand(out_features, requires_grad=True)
        else:
            self.bias = None

    def forward(self, input_tensor):
        output = input_tensor @ self.weight.transpose()
        if self.bias is not None:
            output = output + self.bias
        return output

class ReLU(Module):
    """
    Applies the Rectified Linear Unit function element-wise.
    """
    def forward(self, input_tensor):
        # Using a mask to apply relu and to calculate gradients
        mask = (input_tensor.data > 0)
        out = Tensor(input_tensor.data * mask, requires_grad=input_tensor.requires_grad)

        if out.requires_grad:
            out._prev = {input_tensor}
            def _backward():
                if input_tensor.requires_grad:
                    input_tensor.grad = (input_tensor.grad if input_tensor.grad is not None else 0) + mask * out.grad
            out._backward = _backward
            
        return out
