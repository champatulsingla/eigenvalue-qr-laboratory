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




# Augastya








# Ayush








#Baleshwar




