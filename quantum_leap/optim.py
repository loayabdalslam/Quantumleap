class SGD:
    """
    Implements stochastic gradient descent (optionally with momentum).
    """
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step.
        """
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Clears the gradients of all optimized tensors.
        """
        for p in self.params:
            p.grad = None
