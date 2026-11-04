import eigen_lab, test

import numpy as np
from scipy.linalg import hilbert

if __name__=="__main__":
  obj=test.Testing()

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
  obj1=test.Testing()
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
