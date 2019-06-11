import numpy as np
import pandas as pd
def gradient_descent(X, Y, B, alpha, iterations):
    m = len(Y)
    for iteration in range(iterations):
        AB=[]
        for i in range(len(B)):
            AB.append(abs(B[i]))
        loss = np.dot(X,B) - Y + sum(AB)
        #Gradient Calculation
        gradient = np.dot(loss,X) / m
        #Weight Update
        B = B - alpha * gradient     
    return B
X=[(1,6,3),(4,5,6),(7,8,9),(9,2,7),(5,6,3)]
Y=[8,3,9,8,3]
B=[1,1,1]
alp=0.0001
newB= gradient_descent(X, Y, B, alp,100000)
print(newB)