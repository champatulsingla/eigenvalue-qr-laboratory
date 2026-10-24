# Atul






# Augastya
import numpy as np


class QRFactorization:
    def __init__(self, A):
        self.A=A

    def gramschmidt(self):
            
        n_columns=len(self.A[0])
        n_rows=len(self.A)
        q=np.zeros((n_rows,n_columns),dtype=float)
        for j in range(n_columns):
            s=np.zeros((n_rows),dtype=float)
            for i in range(j):                   
                s=s+(q[:,i]@self.A[:,j])*q[:,i]
            u=self.A[:,j]-s
            q[:,j]=(u/np.linalg.norm(u))
        r=np.zeros((n_columns,n_columns),dtype=float)
        for i in range(n_columns):
            for j in range(n_columns):
                r[i][j]=q[:,i]@self.A[:,j]
        return q,r







# Ayush








#Baleshwar




