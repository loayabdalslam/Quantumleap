import numpy as np
from .parallel import parallel_matmul

# --- Constants ---
# Use parallel matmul for matrices where the product of dimensions exceeds this threshold
PARALLEL_THRESHOLD = 10000

def _unbroadcast(grad, target_shape):
    """
    Un-does broadcasting by summing gradients over the broadcasted dimensions.
    """
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)
    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if grad_dim > target_dim and target_dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    """
    A multi-dimensional matrix containing elements of a single data type.
    This is the fundamental data structure of the QuantumLeap framework.
    """
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            # Ensure data is a float32 numpy array by default
            data = np.array(data, dtype=np.float32)
        
        self.data = data
        self.requires_grad = requires_grad
        
        # Gradients and graph structure for autograd
        self.grad = None # Will be a numpy array of gradients
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        """
        Computes the gradient of this tensor with respect to all its dependencies.
        Performs a topological sort of the graph and applies the chain rule.
        """
        # Build a topological sort of the computation graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # Initialize the gradient of the final node to 1.0
        self.grad = np.ones(self.shape, dtype=self.dtype)

        # Propagate gradients backwards through the graph
        for v in reversed(topo):
            v._backward()

    @property
    def shape(self):
        """Returns the shape of the tensor."""
        return self.data.shape

    @property
    def dtype(self):
        """Returns the data type of the tensor."""
        return self.data.dtype

    def __repr__(self):
        """Provides a string representation of the tensor."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # --- High-Performance Math Operations ---

    def __add__(self, other):
        """Element-wise addition."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad))
        
        if out.requires_grad:
            out._prev = {self, other}
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad if self.grad is not None else 0) + _unbroadcast(out.grad, self.shape)
                if other.requires_grad:
                    other.grad = (other.grad if other.grad is not None else 0) + _unbroadcast(out.grad, other.shape)
            out._backward = _backward
            
        return out

    def __mul__(self, other):
        """Element-wise multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad))
        
        if out.requires_grad:
            out._prev = {self, other}
            def _backward():
                if self.requires_grad:
                    grad_for_self = other.data * out.grad
                    self.grad = (self.grad if self.grad is not None else 0) + _unbroadcast(grad_for_self, self.shape)
                if other.requires_grad:
                    grad_for_other = self.data * out.grad
                    other.grad = (other.grad if other.grad is not None else 0) + _unbroadcast(grad_for_other, other.shape)
            out._backward = _backward
            
        return out

    def __matmul__(self, other):
        """Matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # --- Dynamic Execution ---
        # Decide whether to use parallel or serial matmul
        if self.data.size * other.data.size > PARALLEL_THRESHOLD:
            result_data = parallel_matmul(self.data, other.data)
        else:
            result_data = self.data @ other.data
            
        out = Tensor(result_data, requires_grad=(self.requires_grad or other.requires_grad))
        
        if out.requires_grad:
            out._prev = {self, other}
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad if self.grad is not None else 0) + out.grad @ other.data.T
                if other.requires_grad:
                    other.grad = (other.grad if other.grad is not None else 0) + self.data.T @ out.grad
            out._backward = _backward

        return out

    def sum(self):
        """Computes the sum of all elements in the tensor."""
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad if self.grad is not None else 0) + np.ones_like(self.data) * out.grad
            out._backward = _backward
            
        return out
        
    def mean(self):
        """Computes the mean of all elements in the tensor."""
        total_sum = self.sum()
        num_elements = self.data.size
        # Re-use existing ops to maintain the graph
        out = total_sum * (1.0 / num_elements)
        return out
        
    def transpose(self):
        """Transposes the tensor."""
        out = Tensor(self.data.T, requires_grad=self.requires_grad)

        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad if self.grad is not None else 0) + out.grad.T
            out._backward = _backward
            
        return out

# --- Tensor Creation Helper Functions ---

def zeros(*shape, requires_grad=False):
    """Creates a tensor filled with zeros."""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)

def ones(*shape, requires_grad=False):
    """Creates a tensor filled with ones."""
    return Tensor(np.ones(shape), requires_grad=requires_grad)

def rand(*shape, requires_grad=False):
    """Creates a tensor with random values from a uniform distribution over [0, 1)."""
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)

