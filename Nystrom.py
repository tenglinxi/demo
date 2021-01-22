from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from scipy import linalg
import time
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


if __name__ == '__main__':
    num = 1000
    yy = []
    yyt = []
    for ui in range(num):
        """
        生成数据，设置参数
        """
        n_samples = 4000
        X, y = make_classification(n_samples=n_samples)
        #print("X.shape",X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        n_samples, n_features = X_train.shape
        gamma = 1.0 / n_features


        """
        计算 使用RBF核函数进行分类需要的时间
        """
        clf = svm.SVC(kernel='precomputed')
        start = time.perf_counter()
        gram = rbf_kernel(X_train, X_train, gamma=gamma)
        clf.fit(gram, y_train)
        gram_test = np.dot(X_test, X_train.T)
        result = clf.predict(gram_test)
        end = time.perf_counter()


        """
        输出结果
        """
        #print(zero_one_loss(result - y_test))
        #print("rbf run time is : %.03f seconds" % (end - start), '\n')


        """
        从训练集中随机抽取q=20行数据
        """
        rng = np.random.RandomState(seed=123)
        idx = rng.choice(n_samples, size=20)
        X_train_idx = X_train[idx, :]


        """
        计算选取的20行数据的rbf_kernel K,再对K进行SVD分解得到左奇异向量ui_m
        """
        start = time.perf_counter()
        K = rbf_kernel(X_train_idx, X_train_idx, gamma=gamma)
        ui_m, s, vt = linalg.svd(K, full_matrices=False)


        """
        ui_n为X与20行数据进行rbf_kernel再乘以ui_m, 得到K_{n,m}
        new_K 为K_{n,m}k_{m,m}^{-1}K_{m,n}
        """
        ui_n = rbf_kernel(X_train, X_train_idx, gamma=gamma).dot(ui_m)
        new_K = (ui_n).dot(np.diag(1 / np.sqrt(s))).dot(ui_n.T)
        gram = new_K
        clf.fit(gram, y_train)
        gram_test = np.dot(X_test, X_train.T)
        result = clf.predict(gram_test)
        end = time.perf_counter()
        #print(zero_one_loss(y_test , result))
        #print("nystrom run time is : %.03f seconds" % (end - start))

        yy.append(zero_one_loss(y_test, result))
        yyt.append(end - start)

    x = range(len(yy))

    plt.subplot(2, 1, 1)
    plt.plot(x, yy)
    plt.title('Nystrom method ')
    plt.ylabel("每次实验的平方损失")
    plt.xlabel("实验次数")
    # ui = "平均误差率为：," +str(round(sum(yy)/num,3))
    # plt.text(0.1,0.1,ui)

    plt.subplot(2, 1, 2)
    plt.plot(x, yyt)
    plt.title('Nystrom method ')
    plt.ylabel("每次实验的时间")
    plt.xlabel("实验次数")
    # ui = "平均误时间为：," + str(round(sum(yyt) / num, 3))
    # plt.text(0.1, max(yyt), ui)

    plt.show()
    print("平均误差率为：", round(sum(yy) / num, 3))
    print("平均误时间为：,", str(round(sum(yyt) / num, 3)))
















"""

gamma = 1.0
n = 1500#X.shape[0]
P = np.array([1/1500]*1500) #P0

print(P[0])
s = 10
l = 100

temp = range(1,1501)
ans = []   #R

def SAMPLE_ADAPTIVE():
    for i in range(s):
        ans.append( np.random.choice(temp, p = P.ravel()) )
    t = l/s - 1
    for i in range(t):
        P[i] = UPDATE_PROBABILITY_FULL(ans)
        for j in range(s):
            ans.append( np.random.choice(temp, p = P.ravel()) )
    return ans

def UPDATE_PROBABILITY_FULL(R):
    C = rbf_kernel(R,R,gamma=gamma)
    UC, sigma, VT = np.linalg.svd(C)
    K = C
    E = K - UC.dot(UC.T).dot(K)
    temp_p = np.zeros((1500,1))
    for j in range(n):
        if j in ans:
            temp_p[j] = 0
        else:
            temp_p[j] = np.linalg.norm(E[j], ord=2)
    P = temp_p
    return P

"""

