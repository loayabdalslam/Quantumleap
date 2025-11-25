import numpy as np
import numexpr as ne

class Chip:
    """
    The central "brain" of the QuantumLeap framework.
    It manages and executes all computational operations, using NumExpr for
    optimized element-wise calculations.
    """
    def execute(self, op_name, *inputs):
        """
        Executes an operation using the most efficient available backend.
        """
        if op_name == 'add':
            # Use NumExpr for a potential speedup on large arrays
            return ne.evaluate('a + b', local_dict={'a': inputs[0], 'b': inputs[1]})
        elif op_name == 'mul':
            # Use NumExpr for a potential speedup on large arrays
            return ne.evaluate('a * b', local_dict={'a': inputs[0], 'b': inputs[1]})
        elif op_name == 'matmul':
            # Matmul is not an element-wise operation, so we stick with NumPy's
            # highly optimized BLAS implementation.
            return inputs[0] @ inputs[1]
        else:
            raise NotImplementedError(f"Operation '{op_name}' is not supported by the Chip.")

# --- Global Chip Instance ---
CHIP = Chip()
