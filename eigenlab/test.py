import eigen_lab
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
import numpy as np
class Testing:

  def __init__(self, random_state=2026):
    self.random_state=random_state

  
  def testQR_Factorization(self, A=None, dim=(5,3,3), method='modified_gramschmidt', reorthogonalize=False, tol=1e-12):
    if A is None:
      rng=np.random.default_rng(seed=self.random_state)
      A=rng.random(dim, dtype=float)

    Q=[0]*len(A)
    R=[0]*len(A)
    reconstruction_errors=np.zeros((len(A)))
    orthogonality_errors=np.zeros((len(A)))
    triangularity_errors=np.zeros((len(A)))

    if method=='gramschmidt':
      for i in range(0,len(A)):
        objQRF=eigen_lab.QRFactorization(A[i])
        Q[i], R[i]=objQRF.gramschmidt(tol=tol)
        reconstruction_errors[i]=objQRF.reconstruction(Q[i],R[i])
        orthogonality_errors[i]=objQRF.orthogonality(Q[i])
        triangularity_errors[i]=objQRF.triangularity(Q[i])


    elif method=='modified_gramschmidt':
      for i in range(0,len(A)):
        objQRF=eigen_lab.QRFactorization(A[i])
        Q[i], R[i]=objQRF.modified_gramschmidt(reorthogonalize=reorthogonalize, tol=tol)
        reconstruction_errors[i]=objQRF.reconstruction(Q[i],R[i])
        orthogonality_errors[i]=objQRF.orthogonality(Q[i])
        triangularity_errors[i]=objQRF.triangularity(Q[i])


    else:
      raise ValueError("Invalid method")
    return Q, R, reconstruction_errors, orthogonality_errors, triangularity_errors
    


    







  def test_PowerMethod(self):
    pass





  def testQR_iteration(self,A,max_iter,tol):
    test_object = eigen_lab.QR_Iteration(A,max_iter,tol)
    try:
      test_object.iterate()
      print("convergence is : ",test_object.converged)
      test_object.plot_convergence()
      
    except ValueError as e:
      print("QR iteration stopped",e)



