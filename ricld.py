import numpy as np
def calc(X,s,n,m):
    de=0.0
    for q in range(m-s+1):
        sum=0.0
        x2=np.zeros((n,n), dtype = float)
        for i in range(n):
            for j in range(n):
                x2[i][j] = X[i][j+q]
        z=np.linalg.eigvals(x2)
        print(z)
        a=np.amax(z)**2
        de=max(de,a)
    return de
n=3
m=3
X=[]
s=n
for i in range(n):
    a =[]
    for j in range(m):
        a.append(float(input()))
    X.append(a)
de=calc(X,s,n,m)
if de<1 and de>0:
    print(" matrix A satisfies the s-restricted isometry property with restricted isometry constant δk=", de)
else:
    print(" matrix A doesn't satisfy the s-restricted isometry property")