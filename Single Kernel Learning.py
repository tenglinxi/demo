

from sklearn import datasets
import numpy as np
from scipy.special import erfinv


m=20
n=500
X,Y = datasets.make_regression(n_samples=n, n_features=m)

X = X.reshape(-1,n)
Y = Y.reshape(n,-1)

d = int(m/2) # 10
r = 300 #

sigma = np.ones((1,r))

def get_psi_sigma_x():
    psi_sigma_x = []

    for i in range(d):
        bi = np.random.uniform(0, 1)
        wi = np.random.uniform(0, 1, (m, r))
        hwij = np.sqrt(2) * erfinv(2 * wi - 1)

        num = np.random.randint(1,n,(r,1))
        xj = X[:,num].reshape(-1,r)

        temp = np.sqrt(2/d) * np.cos(np.sum(sigma.dot(hwij.T).dot(xj)) + 2*np.pi*bi )
        psi_sigma_x.append(temp)
    psi_sigma_x = np.array(psi_sigma_x).reshape(-1,1)
    #print(psi_sigma_x.shape)
    return psi_sigma_x

def get_psi_sigma_X():
    ans = np.zeros((d,n))
    for i in range(n):
        ans[:,i:i+1] = get_psi_sigma_x()
    #print(ans.shape)
    return ans

def get_frac_psisigmax_sigmaj():
    frac_psisigmax_sigmaj = []

    for i in range(d):
        bi = np.random.uniform(0, 1)
        wi = np.random.uniform(0, 1, (m, r))
        hwij = np.sqrt(2) * erfinv(2 * wi - 1)

        num = np.random.randint(1, n, (r, 1))
        xj = X[:, num].reshape(-1, r)

        hwijTxj = hwij.T.dot(xj)
        #print(hwijTxj.shape)
        temp = -1 * np.sqrt(2 / d) * hwijTxj[i,i] * np.sin(np.sum(sigma.dot(hwij.T).dot(xj)) + 2 * np.pi * bi)

        frac_psisigmax_sigmaj.append(temp)

    frac_psisigmax_sigmaj = np.array(frac_psisigmax_sigmaj).reshape(-1, 1)
    #print(frac_psisigmax_sigmaj.shape)
    return frac_psisigmax_sigmaj

def get_frac_psisigmaX_sigmaj():
    ans = np.zeros((d, n))
    for i in range(n):
        ans[:, i:i + 1] = get_frac_psisigmax_sigmaj()
    # print(ans.shape)
    return ans









lam = 1.0
step = 0.0005
new_sigma = []
psi_sigma_X = get_psi_sigma_X()
frac_psisigmaX_sigmaj = get_frac_psisigmaX_sigmaj()
w = np.linalg.inv(psi_sigma_X.dot(psi_sigma_X.T) + lam * np.identity(d)).dot(psi_sigma_X).dot(Y)
for i in range(r):

    Q = psi_sigma_X.dot(psi_sigma_X.T) + lam * np.identity(d)
    frac_Q_sigmaj = frac_psisigmaX_sigmaj.dot(psi_sigma_X.T) + psi_sigma_X.dot(frac_psisigmaX_sigmaj.T)

    frac_w_sigmaj = -1 * np.linalg.inv(Q).dot(frac_Q_sigmaj).dot(np.linalg.inv(Q)).dot(psi_sigma_X).dot(
        Y) + np.linalg.inv(Q).dot(frac_psisigmaX_sigmaj).dot(Y)

    frac_f_sigmaj = (psi_sigma_X.T.dot(w) - Y).T
    frac_f_sigmaj = frac_f_sigmaj.dot(frac_psisigmaX_sigmaj.T.dot(w) + psi_sigma_X.T.dot(frac_w_sigmaj))
    frac_f_sigmaj = frac_f_sigmaj + 2 * lam * w.T.dot(frac_w_sigmaj) + 2 * sigma[0][i]
    new_sigma.append(sigma[0][i]-step*frac_f_sigmaj)


new_sigma = np.array(new_sigma).reshape(1,r)
sigma = new_sigma
#print(sigma)

#print(psi_sigma_X.T.dot(w)[0] ,Y[0])

def get_f(sigma):
    f = np.power(np.linalg.norm(psi_sigma_X.T.dot(w) - Y), 2) / 2
    #print(f)
    f = f + lam * (np.power(np.linalg.norm(w), 2))
    f = f + np.power(np.linalg.norm(sigma), 2)
    return f



print("sigma 初始值全为1时，损失函数为",get_f(np.ones((1,r))))
print("经过1次梯度下降后，损失函数为",get_f(sigma))

"""
有问题
"""























"""
psi_sigma_X = get_psi_sigma_X()
frac_psisigmaX_sigmaj = get_frac_psisigmaX_sigmaj()

w = np.linalg.inv(psi_sigma_X.dot(psi_sigma_X.T) + lam*np.identity(d)).dot(psi_sigma_X).dot(Y)


Q = psi_sigma_X.dot(psi_sigma_X.T) + lam*np.identity(d)
frac_Q_sigmaj = frac_psisigmaX_sigmaj.dot(psi_sigma_X.T) + psi_sigma_X.dot(frac_psisigmaX_sigmaj.T)

frac_w_sigmaj = -1*np.linalg.inv(Q).dot(frac_Q_sigmaj).dot(np.linalg.inv(Q)).dot(psi_sigma_X).dot(Y) + np.linalg.inv(Q).dot(frac_psisigmaX_sigmaj).dot(Y)


frac_f_sigmaj = (psi_sigma_X.T.dot(w) - Y).T
frac_f_sigmaj = frac_f_sigmaj.dot(frac_psisigmaX_sigmaj.T.dot(w) + psi_sigma_X.T.dot(frac_w_sigmaj))
frac_f_sigmaj = frac_f_sigmaj + 2*lam*w.T.dot(frac_w_sigmaj) + 2*sigma[0]
"""
