from scipy import stats
import numpy as np
import math

xk = np.arange(1000)
m = 200
p = []
for i in range(1,1001):
    p.append(1/(pow(2,i)))

p = np.array(p)

custm = stats.rv_discrete(name='custm', values=(xk, p))



pp = [1/2,1/2]
xx = [-1,1]
aa = stats.rv_discrete(name='aa', values=(xx, pp))


#f(x)=e^x
def compute_an(n):
    return 1#1/math.factorial(n)




D = 150
z = []






from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=1000, n_features=m)
print(X.shape,y.shape)


def compute_wj():
    temp = []
    for i in range(m):
        temp.append(aa.rvs())
    temp = np.array(temp)

    temp = np.array(temp).reshape(m,1)
    return temp


x = X[0,:].reshape(m,1)

ans = []
#print(custm.rvs(),custm.rvs(),custm.rvs(),custm.rvs(),custm.rvs(),custm.rvs(),custm.rvs(),custm.rvs(),custm.rvs())
p = 10
for i in range(D):
    N = custm.rvs()


    if N==0:
        temp = 0#np.sqrt(compute_an(N)*np.power(p,N+1))*np.sum(x)
        #print(temp)
        ans.append(temp)
        continue


    temp = np.sqrt(compute_an(N)*np.power(p,N+1))
    #print("temp",temp)

    for j in range(N):
        wj = compute_wj()
        temp = temp * (wj.T.dot(x))
    #print(temp)
    ans.append(temp)

new_x = np.array(ans)
new_x = new_x.reshape(150,1)
new_x =new_x/np.sqrt(D)
#print(new_x[0])
#print(new_x.shape)
# print(x.T.dot(x))
#


print(new_x.T.dot(new_x))
print(1/(1-x.T.dot(x)))


#print(x.shape,new_x.shape)


def compute_newx(x):

    ans = []
    for i in range(D):
        N = custm.rvs()

        if N == 0:
            temp = np.sqrt(compute_an(N)*np.power(p,N+1))*np.sum(x)
            # print(temp)
            ans.append(temp)
            continue

        temp = np.sqrt(compute_an(N) * np.power(p, N + 1))
        # print("temp",temp)

        for j in range(N):
            wj = compute_wj()
            temp = temp * (wj.T.dot(x))
        ans.append(temp)
    ans = np.array(ans)
    ans = ans/D
    return ans

x = X[0,:].reshape(m,1)
y = X[1,:].reshape(m,1)

zx = compute_newx(x)
zy = compute_newx(y)

print(zx.T.dot(zy))
print(1/(1-(x.T.dot(y))))
