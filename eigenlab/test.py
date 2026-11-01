try:
  from . import eigen_lab

except ImportError:
  import eigen_lab

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





  def testQR_iteration(self):
    pass



if __name__=="__main__":
  obj=Testing()

  #Experiment 1: Passing 5 random 3X3 matrices with seed=2026
  results=obj.testQR_Factorization()

  print("Orthogonalization errors:",results[3])
  print("Reconstruction errors:", results[2])
  print("Triangularity errors:", results[4])


  #Experiment 2: Passing a rank deficient matrix
  A=np.array([[1,2,3],[4,5,6],[5,7,9]], dtype=float)
  try:
    q,r=eigen_lab.QRFactorization(A).modified_gramschmidt()
    print("Failed to detect rank deficient matrix.")
  except ValueError:
    print("Rank deficient matrix detected successfully")



  #Experiment 3: Hilbert matrices
  obj1=Testing()
  try:
    result_gs=obj1.testQR_Factorization(A=np.array([hilbert(7)], dtype=float), method='gramschmidt')
    print("Orthogonalization error for gram schmidt:",result_gs[3])
    print("Reconstruction error for gram schmidt:", result_gs[2])
    print("Triangularity error for gram schmidt:", result_gs[4])
  except ValueError:
    print("Gramschmidt process broke down")
  try:
    result_mgs=obj1.testQR_Factorization(A=np.array([hilbert(7)], dtype=float), method='modified_gramschmidt')
    print("Orthogonalization error for modified gram schmidt:",result_mgs[3])
    print("Reconstruction error for modified gram schmidt:", result_mgs[2])
    print("Triangularity error for modified gram schmidt:", result_mgs[4])
  except ValueError:
    print("Modified Gramschmidt process broke down")
  try:
    result_re_mgs=obj1.testQR_Factorization(A=np.array([hilbert(7)], dtype=float), method='modified_gramschmidt', reorthogonalize=True)
    print("Orthogonalization error for modified gram schmidt with reorthogonalization:",result_re_mgs[3])
    print("Reconstruction error for modified gram schmidt with reorthogonalization:", result_re_mgs[2])
    print("Triangularity error for modified gram schmidt with reorthogonalization:", result_re_mgs[4])  
  except ValueError:
    print("Modified Gramschmidt process with reorthogonalization broke down")
    
  
  
  
