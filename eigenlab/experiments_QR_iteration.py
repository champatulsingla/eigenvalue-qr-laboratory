import eigen_lab, test
import numpy as np
import matplotlib.pyplot
#exp - 1 -> symmetric matrix 2 x 2
if __name__=="__main__":
    A = np.array([[2., 1.],
                  [1., 3.]])
    exp1 = test.Testing()
    exp1.testQR_iteration(A,1000,1e-8)
    # exp - 2 -> symmetric matrix 3 x 3
    A = A = np.array([[4., 1., 0.],
                  [1., 3., 1.],
                  [0., 1., 2.]])
    exp2 = test.Testing()
    exp2.testQR_iteration(A,1000,1e-8)
    #exp - 3 -> symmetric matrix 4 x 4
    A = np.array([[ 2., -1.,  0.,  0.],
                  [-1.,  2., -1.,  0.],
                  [ 0., -1.,  2., -1.],
                  [ 0.,  0., -1.,  2.]])
    
    exp3 = test.Testing()
    exp3.testQR_iteration(A,1000,1e-8)
    #exp 4  -> Non - symetric matrix with real eigen values
    A = np.array([[4., 1., 0.],
                  [2., 3., 1.],
                  [0., 1., 1.]])
    exp4 = test.Testing()
    exp4.testQR_iteration(A,1000,1e-8)
    # exp 5 -> Non - Symmetric matrix with complex eigen values -- It will fail
    A = np.array([[1., -2., 0.],
                  [2.,  1., 0.],
                  [1.,  0., 3.]])
    exp5 = test.Testing()
    exp5.testQR_iteration(A,1000,1e-8)
