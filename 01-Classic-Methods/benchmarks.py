import numpy as np

class BenchmarkFunction:
    def __init__(self, name, bounds):
        self.name = name
        self.bounds = bounds  

    def __call__(self, x, y):
        raise NotImplementedError

    def gradient(self, x, y):
        raise NotImplementedError

    def hessian(self, x, y):
        raise NotImplementedError
    
    def get_mesh(self, resolution=100):
            """Generates X, Y, Z data for plotting."""
            x = np.linspace(self.bounds[0], self.bounds[1], resolution)
            y = np.linspace(self.bounds[0], self.bounds[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = self.__call__(X, Y)
            return X, Y, Z
    
class Sphere(BenchmarkFunction):
    """
    Implementation of the Sphere function: f(x, y) = x^2 + y^2
    Global minimum is at f(0, 0) = 0.
    """
    def __init__(self):
        # Setting standard bounds for visualization
        super().__init__(name="Sphere", bounds=(-5.0, 5.0))

    def __call__(self, x, y):
        # f(x, y) = x^2 + y^2
        return x**2 + y**2

    def gradient(self, x, y):
        """
        Returns the gradient vector [df/dx, df/dy]
        """
        df_dx = 2 * x
        df_dy = 2 * y
        return np.array([df_dx, df_dy])

    def hessian(self, x, y):
        """
        Returns the 2x2 Hessian matrix:
        [[d2f/dx2, d2f/dxdy],
         [d2f/dydx, d2f/dy2]]
        """
        # For Sphere, the Hessian is always a constant matrix:
        # d2f/dx2 = 2, d2f/dy2 = 2, mixed partials = 0
        h_xx = 2.0
        h_yy = 2.0
        h_xy = 0.0
        return np.array([[h_xx, h_xy], 
                         [h_xy, h_yy]])

# Quick test to verify calculations
if __name__ == "__main__":
    sphere = Sphere()
    test_point = (1.0, 2.0)
    print(f"Value at {test_point}: {sphere(*test_point)}")
    print(f"Gradient at {test_point}: {sphere.gradient(*test_point)}")
    print(f"Hessian at {test_point}:\n{sphere.hessian(*test_point)}")


class Rosenbrock(BenchmarkFunction):
    """
    Implementation of the Rosenbrock function (Banana function).
    f(x, y) = (a - x)^2 + b * (y - x^2)^2
    Standard parameters: a = 1, b = 100
    Global minimum is at f(a, a^2) = 0, usually (1, 1).
    """
    def __init__(self, a=1.0, b=100.0):
        # Setting bounds slightly wider to see the 'banana' shape clearly
        super().__init__(name="Rosenbrock", bounds=(-2.0, 2.0))
        self.a = a
        self.b = b

    def __call__(self, x, y):
        # f(x, y) = (a - x)^2 + b * (y - x^2)^2
        return (self.a - x)**2 + self.b * (y - x**2)**2

    def gradient(self, x, y):
        """
        Returns the gradient vector [df/dx, df/dy]
        """
        df_dx = -2 * (self.a - x) - 4 * self.b * x * (y - x**2)
        df_dy = 2 * self.b * (y - x**2)
        return np.array([df_dx, df_dy])

    def hessian(self, x, y):
        """
        Returns the 2x2 Hessian matrix.
        Calculated as the second-order partial derivatives.
        """
        # d2f/dx2
        h_xx = 2 - 4 * self.b * (y - x**2) + 8 * self.b * (x**2)
        # d2f/dy2
        h_yy = 2 * self.b
        # d2f/dxdy and d2f/dydx (Symmetric)
        h_xy = -4 * self.b * x
        
        return np.array([[h_xx, h_xy], 
                         [h_xy, h_yy]])

# Quick test to verify the global minimum at (1, 1)
if __name__ == "__main__":
    rosen = Rosenbrock()
    min_point = (1.0, 1.0)
    print(f"Value at global minimum {min_point}: {rosen(*min_point)}")
    print(f"Gradient at global minimum: {rosen.gradient(*min_point)}")
    print(f"Hessian at global minimum:\n{rosen.hessian(*min_point)}")


class Rastrigin(BenchmarkFunction):
    """
    Implementation of the Rastrigin function.
    f(x, y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
    Global minimum is at f(0, 0) = 0.
    This function is highly multi-modal with many local minima.
    """
    def __init__(self):
        # Bounds are typically set to [-5.12, 5.12] for this function
        super().__init__(name="Rastrigin", bounds=(-5.12, 5.12))

    def __call__(self, x, y):
        # f(x, y) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)) where A=10, n=2
        return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + \
                    (y**2 - 10 * np.cos(2 * np.pi * y))

    def gradient(self, x, y):
        """
        Returns the gradient vector [df/dx, df/dy]
        df/dx = 2x + 20*pi*sin(2*pi*x)
        """
        df_dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
        df_dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
        return np.array([df_dx, df_dy])

    def hessian(self, x, y):
        """
        Returns the 2x2 Hessian matrix.
        d2f/dx2 = 2 + 40*pi^2*cos(2*pi*x)
        d2f/dxdy = 0 (Separable function)
        """
        h_xx = 2 + 40 * (np.pi**2) * np.cos(2 * np.pi * x)
        h_yy = 2 + 40 * (np.pi**2) * np.cos(2 * np.pi * y)
        h_xy = 0.0
        
        return np.array([[h_xx, h_xy], 
                         [h_xy, h_yy]])

# Quick test to verify the global minimum at (0, 0)
if __name__ == "__main__":
    rastrigin = Rastrigin()
    center = (0.0, 0.0)
    print(f"Value at global minimum {center}: {rastrigin(*center)}")
    print(f"Gradient at global minimum: {rastrigin.gradient(*center)}")
    # At (0,0), Hessian should be large and positive, indicating a sharp minimum
    print(f"Hessian at global minimum:\n{rastrigin.hessian(*center)}")


