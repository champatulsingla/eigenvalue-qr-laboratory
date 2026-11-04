import eigen_lab

# from scipy.linalg import hilbert
import numpy as np
class Testing:

  def __init__(self, random_state=2026):
    self.random_state=random_state

  
  def testQR_Factorization(self, A=None, dim=(5,3,3), method='modified_gramschmidt', reorthogonalize=False, tol=1e-12):
    pass

  def test_PowerMethod(self, A=None, x0=None, tol=1e-8, max_iter=10000):
              # Use a simple matrix with dominant eigenvalue 5 if no matrix is given.
              if A is None:
                  A = np.array([[5.0, 0.0], [0.0, 2.0]])
              if x0 is None:
                  x0 = np.array([1.0, 1.0])
      
              # Run our Power Iteration method.
              method = eigen_lab.PowerIteration(A)
              value, vector, history = method.power_iteration(x0, tol, max_iter)
      
              # Use NumPy to find the reference dominant eigenvalue.
              reference_values = np.linalg.eigvals(A)
              reference_value = reference_values[np.argmax(np.abs(reference_values))]
      
              # Check how accurately the computed pair satisfies A v = lambda v.
              residual = np.linalg.norm(A @ vector - value * vector)
              eigenvalue_error = abs(value - reference_value)
      
              # The method should stop when the eigenpair residual is below tol.
              passed = residual < tol and eigenvalue_error < 10 * tol
      
              print("Power Iteration Test")
              print("Computed eigenvalue:", value)
              print("Reference eigenvalue:", reference_value)
              print("Computed eigenvector:", vector)
              print("Number of iterations:", len(history))
              print("Final residual:", residual)
              print("Eigenvalue error:", eigenvalue_error)
              print("Test result:", "PASS" if passed else "FAIL")
      
              return value, vector, history, residual, passed
  





  def testQR_iteration(self):
    pass

if __name__ == "__main__":
    tester = Testing()
    tester.test_PowerMethod()