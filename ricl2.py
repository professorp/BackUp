import numpy as np
def calc(X,s,n,m):
	de=0.0
	for q in range(m-s+1):
		sum=0.0
		x2=np.zeros((n,n), dtype = float)
		for i in range(n):
			for j in range(n):
				for k in range(q,s+q):
					x2[i][j] = x2[i][j] + X[i][k] * X[j][k]
		z=np.linalg.eigvals(x2)
		print(z)
		a=abs(np.amax(z))
		b=abs(np.amin(z))
		de=max(de,a-1)#,1-b)
	return de
n=3
m=3
X=[]
ma=-1.0
s=2
for i in range(n):
	a =[]
	for j in range(m):
		a.append(float(input()))
	X.append(a)
for i in range(n):
	for j in range(m):
		if X[i][j]>ma:
			ma=X[i][j]
de=calc(X,s,n,m)
if de<1 and de>0:
	de=max(de,ma)
	print(" matrix A satisfies the s-restricted isometry property with restricted isometry constant δk=", de)
else:
	print(" matrix A doesn't satisfy the s-restricted isometry property")