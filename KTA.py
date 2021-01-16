import pandas as pd
import numpy as np
from sklearn.datasets import load_iris #莺尾花数据集
from sklearn.metrics.pairwise import rbf_kernel
from sympy import *
from scipy.linalg import norm
from sklearn.datasets import make_blobs

def kernel_alignment(k1, k2):
    num = np.sum(np.multiply(k1, k2))
    den = (np.linalg.norm(k1))*(np.linalg.norm(k2))
    return num/den

n_samples = 100
x, y = make_blobs(n_samples=n_samples, n_features=5, centers=5,)
x = np.array(x)
y = np.array(y).reshape(-1,1)
K = rbf_kernel(x)
print('original alignment',kernel_alignment(K,np.dot(y,y.T)))
U,sigma,VT = np.linalg.svd(K)

ans = []
for j in range(1,200):
    k = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        v = U[i].reshape(-1, 1)
        a = np.dot(np.dot(v,v.T).T,np.dot(y,y.T))
        a = a.trace()
        #a = np.sum(v*y)
        a = a*(j/100)
        vvT = np.dot(v, v.T)
        k = k+ a*vvT
    ans.append(kernel_alignment(-1*k,np.dot(y,y.T)))
    ans.append(kernel_alignment(1 * k, np.dot(y, y.T)))

print('new alignment',max(ans),ans.index(max(ans)))



a = 0
for i in range(n_samples):
    v = U[i].reshape(-1, 1)
    b = np.dot(np.dot(v, v.T).T, np.dot(y, y.T))
    b = np.square(b.trace())
    a = a + b

a = np.sqrt(a)
print("A(y) = ",a/n_samples)














"""
a1 = kernel_alignment(v1v1T,np.dot(y,y.T))
a2 = kernel_alignment(v2v2T,np.dot(y,y.T))
a3 = kernel_alignment(v3v3T,np.dot(y,y.T))
a4 = kernel_alignment(v4v4T,np.dot(y,y.T))
K = a1*v1v1T + a2*v2v2T + a3*v3v3T + a4*v4v4T
print('alignment',kernel_alignment(K,np.dot(y,y.T)))
"""
"""

sigma = sigma.reshape(-1,1)
fraction = get_a(sigma,0)

vvT = U.dot(VT)
y = U[0].reshape(-1,1)
yyT = y.dot(y.T)


#a1 = U[0]*

#for i in range(1,21):
#a = a* (i/10)
new_K = np.dot(a,vvT)
print(kernel_alignment(new_K,np.dot(y,y.T)))





"""













"""
x, y = make_blobs(n_samples=1500, n_features=10, centers=5,)
x = np.array(x)
y = np.array(y).reshape(-1,1)
K = rbf_kernel(x)
D, V = np.linalg.eig(K)
V,R = np.linalg.qr(V)

a = np.dot(np.dot(V,V.T),np.dot(y,y.T))
new_K = np.dot(a,np.dot(V,V.T))
print(kernel_alignment(new_K,np.dot(y,y.T)))

new_K = np.dot(new_K,new_K)
print(new_K)

new_K = np.sum(new_K)
new_K = np.sqrt(new_K)
"""