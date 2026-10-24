# Atul






# Augastya
import numpy as np


class QRFactorization:
    def __init__(self, A):
        self.A=A
        self.n_columns=len(A[0])
        self.n_rows=len(self.A)


    def gramschmidt(self):
        q=np.zeros((self.n_rows,self.n_columns),dtype=float)
        for j in range(self.n_columns):
            s=np.zeros((self.n_rows),dtype=float)
            for i in range(j):                   
                s=s+(q[:,i]@self.A[:,j])*q[:,i]
            u=self.A[:,j]-s
            q[:,j]=(u/np.linalg.norm(u))
        r=np.zeros((self.n_columns,self.n_columns),dtype=float)
        for i in range(self.n_columns):
            for j in range(self.n_columns):
                r[i][j]=q[:,i]@self.A[:,j]
        return q,r

    def modified_gramschmidt(self):
        q=np.zeros((self.n_rows, self.n_columns))
        r=np.zeros((self.n_columns,self.n_columns))
        for i in range(self.n_columns):
            q[:,i]=self.A[:,i]
        for i in range(self.n_columns):
            r[i,i]=np.linalg.norm(q[:,i])
            q[:,i]=q[:,i]/r[i,i]
            print(r[i,i])
            for j in range(i+1,self.n_columns):
                r[i,j]=q[:,i].T @ q[:,j]
                print(r[i,j])
                q[:,j]=q[:,j]-r[i,j]*q[:,i]
        return q,r

        





# Ayush








#Baleshwar




