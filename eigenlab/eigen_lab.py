import numpy as np
# Atul






# Augastya



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
    





# Ayush








#Baleshwar




