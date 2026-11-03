# Atul
# Power Iteration Method for finding the dominant eigenvalue and eigenvector of a matrix A.
import numpy as np
class PowerIteration:
    def __init__(self, A):
        self.A = np.asarray(A, dtype=np.result_type(A, float))

        # Check whether the matrix is square
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be a square matrix.")
        
    
    def power_iteration(self, x0, tol = 1e-8, max_iter = 10000):
        x = np.asarray(x0, dtype=np.result_type(x0, self.A, float))

        if x.ndim != 1 or  len(x) != self.A.shape[1]:
            raise ValueError("Initial vector x0 must have the same dimension as the number of columns in matrix A.")
        if np.linalg.norm(x) == 0:
            raise ValueError("Initial vector x0 must not be the zero vector.")
        x = x / np.linalg.norm(x)  # normalize the initial vector

        history = []  # to store the history of eigenvalues
            
        y = self.A @ x  # compute the matrix-vector product

        # step : Power iteration loop

        for i in range(max_iter):

            y_norm = np.linalg.norm(y)

            if y_norm < 1e-14:
                history.append((0.0, 0.0))
                return 0.0, x, history
            x_new = y / y_norm    # normalize the new vector

           # Compute A @ x_new once per iteration. 
           # This serves as the next 'y' AND is used for Rayleigh/residual.
           
            y_next = self.A @ x_new
              
            eigenvalue = np.vdot(x_new, y_next)

            # Compute residual using the pre-computed y_next
            residual = np.linalg.norm(y_next - eigenvalue * x_new)

            history.append((eigenvalue, residual))

            # Check convergence using residual
            if residual < tol:
                return eigenvalue, x_new, history
            
            # Setup for next iteration
            x = x_new
            y = y_next

        raise RuntimeError(
            f"Power iteration did not converge within {max_iter} iterations. "
            f"Final residual = {residual:.6e}"
        )




# Augastya








# Ayush








#Baleshwar




