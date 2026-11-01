try:
  from . import eigen_lab

except ImportError:
  import eigen_lab


import numpy as np
class Testing:

  def __init__(self, random_state=2026):
    self.random_state=random_state

  
  def testQR_Factorization(self, A=None, dim=(5,3,3), method='modified_gramschmidt'):
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
        Q[i], R[i]=objQRF.gramschmidt()
        reconstruction_errors[i]=objQRF.reconstruction(Q[i],R[i])
        orthogonality_errors[i]=objQRF.orthogonality(Q[i])
        triangularity_errors[i]=objQRF.triangularity(Q[i])


    elif method=='modified_gramschmidt':
      for i in range(0,len(A)):
        objQRF=eigen_lab.QRFactorization(A[i])
        Q[i], R[i]=objQRF.modified_gramschmidt()
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
  results=obj.testQR_Factorization()
  print("Orthogonalization errors:",results[3])
  print("Reconstruction errors:", results[2])
  print("Triangularity errors:", results[4])

  A=np.array([[1,2,3],[4,5,6],[5,7,9]], dtype=float)
  try:
    q,r=eigen_lab.QRFactorization(A).modified_gramschmidt()
    print("Failed to detect rank deficient matrix.")
  except ValueError:
      print("Rank deficient matrix detected successfully")
