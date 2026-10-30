import eigen_lab
import numpy as np
class Testing:


  
  def testQR_Factorization(self, random_state=2026, A=None, dim=(5,3,3), method='modified_gramschmidt'):
    if A is None:
      rng=np.random.default_rng(seed=random_state)
      A=rng.random(dim, dtype=float)

    objQRF=eigen_lab.QRFactorization(A[0])
    Q=np.zeros((dim[0]))
    R=np.zeros((dim[0]))
    reconstruction_errors=np.zeros((dim[0]))
    orthogonality_errors=np.zeros((dim[0]))


    if method=='gramschmidt':
      Q[0], R[0]=objQRF.gramschmidt()
      reconstruction_errors[0]=objQRF.reconstruction(Q[0],R[0])
      orthogonality_errors[0]=objQRF.orthogonality(Q[0])

      for i in range(1,dim[0]):
        Q[i], R[i]=objQRF.gramschmidt()
        reconstruction_errors[i]=objQRF.reconstruction(Q[i],R[i])
        orthogonality_errors[i]=objQRF.orthogonality(Q[i])


    elif method=='modified_gramschmidt':
      Q[0], R[0]=objQRF.modified_gramschmidt()
      reconstruction_errors[0]=objQRF.reconstruction(Q[0],R[0])
      orthogonality_errors[0]=objQRF.orthogonality(Q[0])

      for i in range(1,dim[0]):
        Q[i], R[i]=objQRF.modified_gramschmidt()
        reconstruction_errors[i]=objQRF.reconstruction(Q[i],R[i])
        orthogonality_errors[i]=objQRF.orthogonality(Q[i])


    else:
      raise ValueError("Invalid method")
    return Q, R, reconstruction_errors, orthogonality_errors



    







  def test_PowerMethod(self):
    pass





  def testQR_iteration(self):
    pass




  
