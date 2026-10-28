# Atul
# Power Iteration Method for finding the dominant eigenvalue and eigenvector of a matrix A.
import numpy as np
class PowerIteration:
    def __init__(self, A):
        self.A = np.array(A, dtype = float)
        # check the matrix is square
        if self.A.shape[0] != self.A.shape[1]:
           raise ValueError("Matrix A must be a square matrix.")
    def rayleigh_quotient(self, x):
        Ax = self.A @ x
        denominator = x.T @ x
        if denominator == 0:
            raise ValueError("Denominator in Rayleigh quotient is zero, cannot compute.")       
        return (x.T @ Ax) / denominator

    def power_iteration(self, x0, tol = 1e-8, max_iter = 1000):
        x = np.array(x0, dtype = float)
        if len(x) != self.A.shape[1]:
            raise ValueError("Initial vector x0 must have the same dimension as the number of columns in matrix A.")
        if np.linalg.norm(x) == 0:
            raise ValueError("Initial vector x0 must not be the zero vector.")
        
        x = x / np.linalg.norm(x)  # normalize the initial vector

        history = []  # to store the history of eigenvalues

        # step : Power iteration loop

        for i in range(max_iter):

            y = self.A @ x  # compute the matrix-vector product
            
            y_norm = np.linalg.norm(y)
            if y_norm == 0:
                raise ValueError("The new vector has zero norm, cannot normalize.")
            x_new = y / y_norm    # normalize the new vector

            # compute the corresponding eigenvalue using the Rayleigh quotient
            eigenvalue = self.rayleigh_quotient(x_new)   

            # Now calculate the change between the new and old eigenvectors
            change = min(np.linalg.norm(x_new - x), np.linalg.norm(-x_new - x))
            residual = np.linalg.norm(np.dot(self.A, x_new) - eigenvalue * x_new)  # compute the residual

            history.append((eigenvalue, change, residual))  # store the eigenvalue, change, and residual
            
            x = x_new  # update the vector for the next iteration 
        # Check for convergence
            if change < tol :
             break

        return eigenvalue, x, history


if __name__ == "__main__":

    # Define the matrix and initial vector
    n = 5
    A = np.random.rand(n, n)
    x0 = np.random.rand(n)

    # Create an object of the PowerIteration class
    obj = PowerIteration(A)

    # Run the power iteration method
    eigenvalue, eigenvector, history = obj.power_iteration(x0)

    # Print the last iteration
    print("\n--- Last Iteration ---")
    print("\nIteration | Eigenvalue | Change | Residual")

    result = len(history)
    value, change, residual = history[-1]

    print(
        f"{result:9d} | {value:10.6f} | "
        f"{change:.3e} | {residual:.3e}"
    )

    print(" Eigenvalue:", eigenvalue)
    print("Corresponding Eigenvector:", eigenvector)
    print("Change:", change)
    print("Residual:", residual)

    # Compare with NumPy's result
    values, vectors = np.linalg.eig(np.array(A, dtype=float))
    index = np.argmax(np.abs(values))

    print("\nNumPy reference eigenvalue:", values[index])




# Augastya








# Ayush








#Baleshwar




