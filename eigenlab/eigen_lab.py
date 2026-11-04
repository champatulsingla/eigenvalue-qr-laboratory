import numpy as np
import matplotlib.pyplot as plt
import warnings
# Power Iteration Method for finding the dominant eigenvalue and eigenvector of a matrix A.


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

            # Check if vector is annihilated by A
            if y_norm < 1e-16:
                # Calculate norm of matrix A to check if A is a zero matrix
                matrix_norm = np.linalg.norm(self.A)
                
                if matrix_norm > 1e-16:
                    warnings.warn(
                        f"Vector x was annihilated by A at iteration {i} (lies in null space). "
                        f"Since matrix A is non-zero, try re-running with a different initial vector x0.",
                        UserWarning
                    )
                else:
                    warnings.warn(
                        f"Matrix A is effectively zero (norm = {matrix_norm:.2e}). "
                        f"Dominant eigenvalue is 0.0.",
                        UserWarning
                    )

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






class QRFactorization:
    def __init__(self, A):
        self.A=np.array(A, dtype=float)
        if self.A.ndim!=2:
            raise ValueError("Please enter 2 dimensional matrix only.")
        self.n_columns=len(A[0])
        self.n_rows=len(self.A)
        


    def gramschmidt(self, tol=1e-8):

        if self.n_columns>self.n_rows:   
            raise ValueError("Number of columns must be less than or equal the number of rows.")    # Handling "fat matrices" where the gramschmidt process breaks down
        
        q=np.zeros((self.n_rows,self.n_columns),dtype=float)

        for j in range(self.n_columns):    # Computing the Q matrix
            s=np.zeros((self.n_rows),dtype=float)
            for i in range(j):                   
                s=s+(q[:,i]@self.A[:,j])*q[:,i]
            u=self.A[:,j]-s
            if np.linalg.norm(u)<tol:    # In case the columns are linearly dependent, norm of vector 'u' will be zero or nearly zero.
                raise ValueError("Matrix has linearly dependent or almost linearly dependent columns")
            q[:,j]=(u/np.linalg.norm(u))
        r=np.zeros((self.n_columns,self.n_columns),dtype=float)
        for i in range(self.n_columns):   # Computing the R matrix
            for j in range(i,self.n_columns):
                r[i][j]=q[:,i]@self.A[:,j]
        
        return q,r

    def modified_gramschmidt(self, reorthogonalize=False, tol=1e-8):
        if self.n_columns>self.n_rows:               
            raise ValueError("Number of columns must be less than or equal the number of rows.")     # Handling "fat matrices" where the gramschmidt process breaks down
        q=np.zeros((self.n_rows, self.n_columns))
        r=np.zeros((self.n_columns,self.n_columns))
        q=self.A.copy()
        for i in range(self.n_columns):         #ith row of R and ith column of Q
            r[i,i]=np.linalg.norm(q[:,i])
            if r[i,i]<tol:                         # In case the columns are linearly dependent, norm of vector r[i,i] will be zero or nearly zero.
                raise ValueError("Matrix has linearly dependent or almost linearly dependent columns")
            q[:,i]=q[:,i]/r[i,i]
            for j in range(i+1,self.n_columns):                                        
                r[i,j]=q[:,i].T @ q[:,j]                            # for each remaining column j: record its coefficient along q_i in R, 
                q[:,j]=q[:,j]-r[i,j]*q[:,i]                         # then subtract that component so column j is orthogonal to q_i
        
        if reorthogonalize and self.orthogonality(q)>tol:            #If computation errors result in a significant loss of orthogonality of matrix Q, we reorthogonalize.
            q2=q.copy()
            r2=np.zeros((self.n_columns,self.n_columns))
            for i in range(self.n_columns):
                r2[i,i]=np.linalg.norm(q2[:,i])
                q2[:,i]=q2[:,i]/r2[i,i]
                for j in range(i+1,self.n_columns):
                    r2[i,j]=q2[:,i].T @ q2[:,j]
                    q2[:,j]=q2[:,j]-r2[i,j]*q2[:,i]
            return q2,r2@r
        else:
            return q,r

    def reconstruction(self, q, r):         # For checking the reconstruction error
        return np.linalg.norm(self.A-q@r) 
    def orthogonality(self,q):               #For checking the orthogonality error of matrix Q
        return np.linalg.norm(q.T@q-np.eye(self.n_columns))
    def triangularity(self, q):               #For checking the upper-triangularity of matrix R
        return np.linalg.norm(np.tril(q.T @ self.A, -1))
    




class QR_Iteration():
    def __init__(self, A, max_iter, tol):
        self.A = np.array(A,dtype=float)
        if self.A.ndim!=2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Expecting a 2D square matrix")
        self.n = self.A.shape[0]
        self.max_iter = max_iter
        self.tol = tol
        self.qr=QRFactorization(A)
        # eigen values calculated using numpy
        ev = np.linalg.eigvals(np.array(A, dtype=float))
        self.np_eigs = ev[np.argsort(-np.abs(ev))]
        # for storing diagonals of every iteration 
        self.diag_history = []     
        self.count = 0
        self.converged = False
    def stopping_Check(self):
            sum = 0
            for i in range(self.n):
                for j in range(self.n):
                    if i > j:
                        sum += (self.A[i][j]**2)
            sum = sum**0.5
            if sum < self.tol:
                return True
            return False
    def iterate(self):
        self.count = 0
        while self.count < self.max_iter and not self.stopping_Check():
            self.count+=1
            Q,R = self.qr.modified_gramschmidt(reorthogonalize=True,tol = self.tol)
            self.A = R@Q
            # updation in qrFactorisation object
            self.qr.A = self.A
            d = np.diag(self.A)
            self.diag_history.append(d[np.argsort(-np.abs(d))])
        self.converged = self.stopping_Check()
        return self.converged
    def plot_convergence(self):
        history = np.array(self.diag_history)
        if len(history) == 0:
            print(" Please call iterate first ")
            return      
        k = np.arange(len(history))
        # iteration start from 1
        k+=1
        fig, axes = plt.subplots(1, self.n, figsize=(5 * self.n, 4))
        axes = np.atleast_1d(axes)
        for i in range(self.n):
            axes[i].plot(k, history[:, i], label=f"QR iteration, diag[{i}]")
            axes[i].axhline(self.np_eigs[i], color="r", linestyle="--",
                            label=f"NumPy eigenvalue = {self.np_eigs[i]:.4f}")
            axes[i].set_ylabel(f"$\\lambda_{i+1}$")
            axes[i].legend(loc="best")
            axes[i].grid(True)
            axes[i].set_xlabel("Iteration")
        fig.suptitle("Convergence of diagonal entries to eigenvalues")
        plt.tight_layout()
        plt.show()
 







#Baleshwar




