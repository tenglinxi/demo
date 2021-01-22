import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy import linalg
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
from scipy import linalg
import time
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
plt.rcParams['font.sans-serif'] = ['SimHei']

def SAMPLE_ADAPTIVE(X_train):
    P = np.array([1 / X_train.shape[0]] * X_train.shape[0])
    P = P.reshape(-1,1)
    for i in range(s):
        ans.append( np.random.choice(temp, p = P.ravel()) )
    t = l/s - 1
    for i in range(int(t)):
        P = UPDATE_PROBABILITY_PARTIAL(ans,X_train,s)
        for j in range(s):
            ans.append( np.random.choice(temp, p = P.ravel()) )
    return ans

def UPDATE_PROBABILITY_FULL(R):
    R = np.array(R)
    R = K[:,R]
    C = rbf_kernel(R,R)
    UC, sigma, VT = np.linalg.svd(C)
    E = K - UC.dot(UC.T).dot(K)
    temp_p = np.zeros((1500,1)).reshape(-1,1)
    for j in range(n):
        #print(np.linalg.norm(E[j], ord=2))
        if j in ans:
            temp_p[j] = 0
        else:
            temp_p[j] = np.linalg.norm(E[j], ord=2)

    P = np.square( temp_p/np.sqrt(np.sum(np.square(temp_p))) )
    return P

def UPDATE_PROBABILITY_PARTIAL(R,X_train,s):
    R = np.array(R)
    temp = R

    C_pie = rbf_kernel(X_train, X_train[R,:])
    k_pie = int(s/2)
    ui_m, s, vt = linalg.svd(C_pie, full_matrices=False)


    new_K = (ui_m[:,0:k_pie]).dot(np.diag(1 / np.sqrt(s[0:k_pie]))).dot(vt[0:k_pie,:])

    C_nys = new_K
    E = C_pie - C_nys
    temp_p = np.zeros((X_train.shape[0], 1)).reshape(-1, 1)
    for j in range(X_train.shape[0]):
        if j in temp:
            temp_p[j] = 0
        else:
            temp_p[j] = np.linalg.norm(E[j], ord=2)
    P = np.square( temp_p/np.sqrt(np.sum(np.square(temp_p))) )

    return P

if __name__ == '__main__':

    yy = []
    yyt = []
    num = 1000
    for i in range(num):
        n = 4000
        X, y = make_classification(n_samples=n)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        ans = []
        s = 5
        l = 20
        temp = np.array(range(0, X_train.shape[0]))

        start = time.perf_counter()
        ans = SAMPLE_ADAPTIVE(X_train)

        X_train_idx = X_train[ans, :]
        K = rbf_kernel(X_train_idx, X_train_idx)
        ui_m, ss, vt = linalg.svd(K, full_matrices=False)
        ui_n = rbf_kernel(X_train, X_train_idx).dot(ui_m)
        new_K = (ui_n).dot(np.diag(1 / np.sqrt(ss))).dot(ui_n.T)

        end = time.perf_counter()
        clf = svm.SVC(kernel='precomputed')
        gram = new_K
        clf.fit(gram, y_train)
        gram_test = np.dot(X_test, X_train.T)
        result = clf.predict(gram_test)

        #print(np.sum(np.square(result - y_test)))
        #print("Adaptive nystrom run time is : %.03f seconds" % (end - start))

        yy.append(zero_one_loss(y_test,result))
        yyt.append(end - start)
    x = range(len(yy))

    plt.subplot(2, 1, 1)
    plt.plot(x, yy)
    plt.title('Adaptive Nystrom Sampling ')
    plt.ylabel("每次实验的平方损失")
    plt.xlabel("实验次数")
    #ui = "平均误差率为：," +str(round(sum(yy)/num,3))
    #plt.text(0.1,0.1,ui)

    plt.subplot(2, 1, 2)
    plt.plot(x, yyt)
    plt.title('Adaptive Nystrom Sampling ')
    plt.ylabel("每次实验的时间")
    plt.xlabel("实验次数")
    #ui = "平均误时间为：," + str(round(sum(yyt) / num, 3))
    #plt.text(0.1, max(yyt), ui)

    plt.show()
    print("平均误差率为：",round(sum(yy)/num,3))
    print("平均误时间为：," , str(round(sum(yyt) / num, 3)))