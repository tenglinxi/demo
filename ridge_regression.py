from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_ridge import KernelRidge
import time

class my_model():
    def __init__(self):
        self.n_samples = 1500
        self.n_features = 4
        self.rf_gamma = 2
        self.D = 10#int(self.n_features/2)
        self.d = self.n_features

    def my_data(self, mode = 'train'):
        X, y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=3,center_box=(10,30))
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=0.1,)
        if mode == 'train':
            return self.train_X,self.train_y
        elif mode == 'test':
            return self.test_X,self.test_y

    def my_RR_omega(self,train_X,train_y):
        """
        计算 ridge regression 的参数
        :param train_X:
        :param train_y:
        :return:
        """
        train_X = np.mat(train_X)
        train_y = np.mat(train_y)

        xTx = train_X.T * train_X
        denom = xTx + np.eye(np.shape(train_X)[1]) * 1.0
        if np.linalg.det(denom) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        omega = denom.I * (train_X.T * train_y)
        return omega


    def my_RR_prediction(self,omega,test_X):
        """
        根据岭回归模型的参数 进行预测
        :param omega:
        :param test_X:
        :return:
        """
        return np.dot(test_X,omega)

    def my_KRR_a(self,train_X,train_y):
        """
        计算kernel ridge regression 的参数
        :param train_X:
        :param train_y:
        :return:
        """
        K = rbf_kernel(train_X)
        a = np.dot(np.linalg.inv(K + np.identity(train_X.shape[0])), train_y)
        return a

    def my_KRR_prediction(self,a,train_X,test_X):
        """
        根据kernel ridge regression模型的参数 进行预测
        :param a:
        :param train_X:
        :param test_X:
        :return:
        """
        new_K = rbf_kernel(train_X,test_X)
        return np.dot(new_K.T,a)

    def my_RR_matrix(self, X1, X2):
        """
        原始数据X1，X2，经过随机特征变换后的两个向量，相乘得到的矩阵。
        :param X1:
        :param X2:
        :return:
        """
        w = np.sqrt(2 * self.rf_gamma) * np.random.normal(size=(self.D, self.d))
        u = 2 * np.pi * np.random.rand(self.D)
        Z1 = np.sqrt(2 / self.D) * np.cos((X1.dot(w.T) + u[np.newaxis, :]))
        Z2 = np.sqrt(2 / self.D) * np.cos((X2.dot(w.T) + u[np.newaxis, :]))
        return np.dot(Z1.T, Z2)

    def my_RR(self, X1):
        """
        原始数据X1 经过随机特征变换后 形成的向量。
        :param X1:
        :return:
        """
        w = np.sqrt(2 * self.rf_gamma) * np.random.normal(size=(self.D, self.d))
        u = 2 * np.pi * np.random.rand(self.D)
        Z1 = np.sqrt(2 / self.D) * np.cos((X1.dot(w.T) + u[np.newaxis, :]))
        return Z1

    def my_RRRR(self, train_X, train_y):
        """
        计算 随机特征岭回归模型 的参数
        :param train_X:
        :param train_y:
        :return:
        """
        K = self.my_RR_matrix(train_X, train_X)
        a = np.dot(np.linalg.inv(K + np.identity(K.shape[0])), self.my_RR(train_X).T)
        a = np.dot(a, train_y)
        return a

    def my_RRRR_prediction(self, a, test_X):
        """
        运用 随机特征岭回归的参数 进行预测
        :param a:
        :param test_X:
        :return:
        """
        pre = np.dot(self.my_RR(test_X), a)
        return pre


if __name__ == '__main__':

    m = my_model()
    train_X, train_y = m.my_data()
    test_X, test_y = m.my_data(mode='test')

    start = time.perf_counter()
    omega = m.my_RR_omega(train_X, train_y)#计算参数
    pre = m.my_RR_prediction(omega, test_X)#进行预测
    end = time.perf_counter()
    print('岭回归的均方误差',np.mean(np.square(pre - test_y)))
    print("岭回归 run time is : %.03f seconds" % (end - start),'\n')

    start = time.perf_counter()
    a = m.my_KRR_a(train_X, train_y)#计算参数
    pre = m.my_KRR_prediction(a,train_X,test_X)#进行预测
    end = time.perf_counter()
    print('核岭回归的均方误差',np.mean((pre-test_y)**2))
    print("核岭回归 run time is : %.03f seconds" % (end - start),'\n')

    start = time.perf_counter()
    a = m.my_RRRR(train_X, train_y)#计算参数
    pre = m.my_RRRR_prediction(a,test_X)#进行预测
    end = time.perf_counter()
    print('随机特征岭回归的均方误差',np.mean(np.square(pre-test_y)))
    print("随机特征岭回归 run time is : %.03f seconds" % (end - start))

