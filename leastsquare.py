import numpy as np
def ls(X,y):
	B=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
	print(B)
	yd=X*B
	return yd
X=[(1,6,3),(4,5,6),(7,8,9),(9,2,7),(5,6,3)]
Y=[8,3,9,8,3]
Z=ls(X,y)
print(Z)
"""
data = pd.read_csv('headbrain.csv')
print(data.shape)
math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values
m = len(math)
x0 = np.ones(m)
X=np.array([x0, math, read]).T
Y=np.array(write)
Z=np.array([1,1,1])
m = len(Y)
cost = np.sum((np.dot(X,newB) - Y) ** 2)/(2 * m)
print(cost)
"""