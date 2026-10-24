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
            if np.linalg.norm(u)<1e-8:
                return None, None
            q[:,j]=(u/np.linalg.norm(u))
        r=np.zeros((self.n_columns,self.n_columns),dtype=float)
        for i in range(self.n_columns):
            for j in range(self.n_columns):
                r[i][j]=q[:,i]@self.A[:,j]
        return q,r

    def modified_gramschmidt(self, reorthogonalize=False, tol=1e-8):
        q=np.zeros((self.n_rows, self.n_columns))
        r=np.zeros((self.n_columns,self.n_columns))
        for i in range(self.n_columns):
            q[:,i]=self.A[:,i]
        for i in range(self.n_columns):
            r[i,i]=np.linalg.norm(q[:,i])
            if r[i,i]<1e-8:
                return None, None
            q[:,i]=q[:,i]/r[i,i]
            for j in range(i+1,self.n_columns):
                r[i,j]=q[:,i].T @ q[:,j]
                q[:,j]=q[:,j]-r[i,j]*q[:,i]
        
        if self.orthogonality(q)>tol and reorthogonalize:
            q2=np.zeros((self.n_rows, self.n_columns))
            r2=np.zeros((self.n_columns,self.n_columns))
            for i in range(self.n_columns):
                q2[:,i]=q[:,i]
            for i in range(self.n_columns):
                r2[i,i]=np.linalg.norm(q2[:,i])
                q2[:,i]=q2[:,i]/r2[i,i]
                for j in range(i+1,self.n_columns):
                    r2[i,j]=q2[:,i].T @ q2[:,j]
                    q2[:,j]=q2[:,j]-r2[i,j]*q2[:,i]
            return q2,r2@r
        else:
            return q,r

    def reconstruction(self, q, r):
        if q is not None:
            return np.linalg.norm(self.A-q@r)
        else: 
            return None
    def orthogonality(self,q):
        if q is not None:
            return np.linalg.norm(q.T@q-np.eye(self.n_columns))
        else:
            return None

if __name__=="__main__":
    a=np.random.rand(100,90)
    obj=QRFactorization(a)
    q,r=obj.gramschmidt()
    q_MGS, r_MGS=obj.modified_gramschmidt(reorthogonalize=True, tol=1e-8)
    o_error=obj.orthogonality(q)
    r_error=obj.reconstruction(q,r)
    r_error_MGS=obj.reconstruction(q_MGS, r_MGS)
    o_error_MGS=obj.orthogonality(q_MGS)
    print(o_error, r_error, o_error_MGS, r_error_MGS)
    





# Ayush








#Baleshwar




