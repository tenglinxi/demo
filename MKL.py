import numpy as np
from sklearn import svm,datasets
from scipy.special import erfinv
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split


class MKL():
    def __init__(self,sigma):
        self.n = 500
        self.m = 20
        self.d = 10
        self.r = 2
        self.X, self.Y = datasets.make_regression(n_samples=self.n, n_features=self.m)

        self.X = self.X.reshape(self.m, self.n)
        self.Y = self.Y.reshape(self.n, 1)

        self.sigma = sigma
        self.lam = 1.0

        self.step = 0.000001

        self.psi_sigma_X = self.compute_psi_sigma_X()
        self.w_sigma = self.compute_w_sigma()
        self.Q = self.compute_Q()


    def compute_psi_sigma_X(self,):
        psi_sigma_X = []

        for i in range(self.d):
            bi = np.random.uniform(0, 1)
            wi = np.random.uniform(0, 1, (self.r, self.n))
            #print(self.sigma)
            hwij = self.sigma * np.sqrt(2) * erfinv(2 * wi - 1)

            xj = self.X[i * self.r:(i + 1) * self.r, :]

            temp = np.sqrt(2 / self.d) * np.cos(np.diag(hwij.T.dot(xj)) + 2 * np.pi * bi)

            psi_sigma_X.append(temp)

        psi_sigma_X = np.array(psi_sigma_X)
        #print(psi_sigma_X.shape)
        return psi_sigma_X

    def compute_w_sigma(self):
        w_sigma = np.linalg.inv(self.psi_sigma_X.dot(self.psi_sigma_X.T) + self.lam * np.identity(self.d)).dot(self.psi_sigma_X).dot(self.Y)
        w_sigma = w_sigma.reshape(-1, 1)
        return w_sigma

    def compute_error_function(self):
        f = np.power(np.linalg.norm(self.psi_sigma_X.T.dot(self.w_sigma) - self.Y), 2) / 2
        f = f + self.lam * (np.power(np.linalg.norm(self.w_sigma), 2))
        #print(f)
        return f


    def compute_frac_psisigmaX_sigmaj(self,num):# 0,1
        frac_psisigmaX_sigmaj = []
        for i in range(self.d):
            bi = np.random.uniform(0, 1)
            wi = np.random.uniform(0, 1, (self.r, self.n))
            hwij = self.sigma * np.sqrt(2) * erfinv(2 * wi - 1)

            xj = self.X[i * self.r:(i + 1) * self.r, :]

            x_j = self.X[num + i * self.r, :].reshape(-1, 1)

            hwij_j = hwij[0, :].reshape(-1, 1)

            # print(hwij_j.shape)
            # print(np.diag(hwij[0,:].dot(x_j)))
            temp = np.sqrt(2 / self.d) * np.diag(hwij_j.dot(x_j.T)) * np.cos(np.diag(hwij.T.dot(xj) + 2 * np.pi * bi))

            # print(temp.shape)

            frac_psisigmaX_sigmaj.append(temp)

        frac_psisigmaX_sigmaj = np.array(frac_psisigmaX_sigmaj)
        return frac_psisigmaX_sigmaj

    def compute_frac_Q_sigmaj(self,frac_psisigmaX_sigmaj):
        frac_Q_sigmaj = frac_psisigmaX_sigmaj.dot(self.psi_sigma_X.T) + self.psi_sigma_X.dot(frac_psisigmaX_sigmaj.T)
        #print(frac_Q_sigmaj.shape)
        return frac_Q_sigmaj



    def compute_Q(self):
        Q = self.psi_sigma_X.dot(self.psi_sigma_X.T) + self.lam * np.identity(self.d)
        return Q

    def compute_frac_wsigma_sigmaj(self,frac_Q_sigmaj,frac_psisigmaX_sigmaj):
        frac_wsigma_sigmaj = - np.linalg.inv(self.Q).dot(frac_Q_sigmaj).dot(np.linalg.inv(self.Q)).dot(self.psi_sigma_X).dot(self.Y)
        frac_wsigma_sigmaj = frac_wsigma_sigmaj + np.linalg.inv(self.Q).dot(frac_psisigmaX_sigmaj).dot(self.Y)
        #print(frac_wsigma_sigmaj.shape)
        return frac_wsigma_sigmaj

    def compute_frac_f_sigmaj(self,frac_psisigmaX_sigmaj,frac_wsigma_sigmaj):
        temp = self.psi_sigma_X.T.dot(self.w_sigma) - self.Y
        temp = temp.T.dot(frac_psisigmaX_sigmaj.T.dot(self.w_sigma) + self.psi_sigma_X.T.dot(frac_wsigma_sigmaj))
        temp = temp + 2 * self.lam * self.w_sigma.T.dot(frac_wsigma_sigmaj)
        #print(temp.shape)
        return temp

    def compute_new_sigma(self):
        new_sigma = np.ones((self.r, self.n))

        frac_psisigmaX_sigmaj = self.compute_frac_psisigmaX_sigmaj(0)
        frac_Q_sigmaj = self.compute_frac_Q_sigmaj(frac_psisigmaX_sigmaj)
        frac_psisigmaX_sigmaj = self.compute_psi_sigma_X()
        frac_wsigma_sigmaj = self.compute_frac_wsigma_sigmaj(frac_Q_sigmaj,frac_psisigmaX_sigmaj)
        frac_f_sigma1 = self.compute_frac_f_sigmaj(frac_psisigmaX_sigmaj,frac_wsigma_sigmaj)
        #print("1111",frac_f_sigma1.shape)
        new_sigma[0, :] = self.sigma[0][0] - self.step * frac_f_sigma1

        frac_psisigmaX_sigmaj = self.compute_frac_psisigmaX_sigmaj(1)
        frac_Q_sigmaj = self.compute_frac_Q_sigmaj(frac_psisigmaX_sigmaj)
        frac_psisigmaX_sigmaj = self.compute_psi_sigma_X()
        frac_wsigma_sigmaj = self.compute_frac_wsigma_sigmaj(frac_Q_sigmaj, frac_psisigmaX_sigmaj)
        frac_f_sigma2 = self.compute_frac_f_sigmaj(frac_psisigmaX_sigmaj, frac_wsigma_sigmaj)
        new_sigma[1, :] = self.sigma[1][0] - self.step * frac_f_sigma2

        return new_sigma

import matplotlib.pyplot as plt
if __name__=='__main__':
    sigma = np.ones((2, 500))
    mkl = MKL(sigma)

    yy = []
    for i in range(100):
        new_sigma = mkl.compute_new_sigma()

        new_mkl = MKL(new_sigma)

        temp = new_mkl.compute_error_function()
        print(temp)
        yy.append(temp)

    plt.plot(yy)
    plt.show()

"""
    sigma = mkl.compute_new_sigma()
    #print(sigma.shape)
    #print(sigma[0][0],[1][0])
    #print(mkl.compute_w_sigma())
    psi_sigma_X = mkl.compute_psi_sigma_X()
    #print(mkl.compute_psi_sigma_X().shape)
    K1 = 1/np.sqrt(2*np.pi)*np.exp(np.power(np.linalg.norm()))
    K2 = rbf_kernel(mkl.X.T,gamma=1.0)

    print(np.sum(K1-K2))
    print(K1[0][0],K2[0][0])
    print(K1[0][1],K2[0][1])
"""