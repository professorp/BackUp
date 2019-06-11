import numpy as np
X=[(1,6,3),(4,5,6),(7,8,9),(9,2,7),(5,6,3)]
Y=[8,3,9,8,3]
Z= np.linalg.lstsq(X, Y, rcond=-1)[0]
print(Z)