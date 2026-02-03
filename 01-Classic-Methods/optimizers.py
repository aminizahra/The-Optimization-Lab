import numpy as np

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.history = [] # To store path for animation

    def solve(self, func, x_start, y_start):
        self.history = [(x_start, y_start)]
        x, y = x_start, y_start
        
        for _ in range(self.n_iterations):
            x, y = self.step(func, x, y)
            self.history.append((x, y))
            
            # Optional: Add convergence check (if gradient is very small)
            
        return np.array(self.history)

    def step(self, func, x, y):
        raise NotImplementedError

class GradientDescent(Optimizer):
    def step(self, func, x, y):
        grad = func.gradient(x, y)
        new_x = x - self.lr * grad[0]
        new_y = y - self.lr * grad[1]
        return new_x, new_y

class NewtonsMethod(Optimizer):
    def step(self, func, x, y):
        grad = func.gradient(x, y)
        hessian = func.hessian(x, y)
        
        try:
            # Solve H * delta = -grad instead of explicit inversion for stability
            delta = np.linalg.solve(hessian, -grad)
            return x + delta[0], y + delta[1]
        except np.linalg.LinAlgError:
            # Fallback to GD if Hessian is singular
            return x - 0.01 * grad[0], y - 0.01 * grad[1]

class VanillaSGD(Optimizer):
    def __init__(self, learning_rate=0.01, n_iterations=100, noise_level=0.1):
        super().__init__(learning_rate, n_iterations)
        self.noise_level = noise_level

    def step(self, func, x, y):
        grad = func.gradient(x, y)
        # Add artificial noise to simulate mini-batch randomness
        noise = np.random.normal(0, self.noise_level, size=2)
        noisy_grad = grad + noise
        
        new_x = x - self.lr * noisy_grad[0]
        new_y = y - self.lr * noisy_grad[1]
        return new_x, new_y