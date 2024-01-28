import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numpy.random import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from random import *
import pickle

from cvxopt import matrix
from cvxopt import solvers

import imageio
from sklearn.decomposition import PCA

solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10


def quadratic_solver(K, y, beta):
    """
    :param K: Kernel matrix K of shape (m,m)
    :param y: array of binary labels {-1, 1} of shape (m,)
    :param beta: ridge regularization coefficient
    :return: optimal alphas of shape (m,)
    """

    m = K.shape[0]

    I = np.identity(m)
    P = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            P[i, j] = y[i] * y[j] * K[i, j] + beta * I[i, j]

    q = -np.ones((m, 1))  # shape(m,1)
    G = -np.eye(m)  # shape(m,m)
    h = 0.0 * np.zeros(m)  # shape (m,)
    A = 1.0 * y.reshape(1, -1)  # shape (1,m)
    b = 0.0  # scalar
    # Quadratic solver
    sol = solvers.qp(P=matrix(P.astype(float)),
                     q=matrix(q.astype(float)),
                     G=matrix(G.astype(float)),
                     h=matrix(h.astype(float)),
                     A=matrix(A.astype(float)),
                     b=matrix(b))
    alphas = np.array(sol['x'])
    alphas = alphas * (np.abs(alphas) > 1e-8)  # zeroing out the small values
    return alphas.reshape(-1)

class LinearKernel(object):

    def __init__(self, **kwargs):
        pass
    def compute_kernel(self, X1, X2):
        """
        Compute the kernel matrix
        @param X1: array of shape (m1, d)
        @param X2: array of shape(m2, d)
        @return: K of shape (m1, m2) where K[i,j] = <X1[i], X2[j]>
        """
        K = np.zeros((X1.shape[0], X2.shape[0]))
        K = np.dot(X1, np.transpose(X2))
        return K


class RadialKernel(object):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma

    def compute_kernel(self, X1, X2):
        """
        Compute the kernel matrix. Hint: computing the squared distances is similar to compute_distances in K-means
        @param X1: array of shape (m1, d)
        @param X2: array of shape(m2, d)
        @return: K of shape (m1,m2) where K[i,j] = K_rad(X1[i],X2[j]) = exp(-gamma * ||X1[i] - X2[j]||^2)
        """
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                K[i][j] = np.exp(-self.gamma * ((np.linalg.norm(X1[i] - X2[j]))**2))
        return K


class PolynomialKernel(object):

    def __init__(self, c, p, **kwargs):
        self.c = c
        self.p = p

    def compute_kernel(self, X1, X2):
        """
        Compute the kernel matrix.
        @param X1: array of shape (m1, d)
        @param X2: array of shape(m2, d)
        @return: K of shape (m1,m2) where K[i,j] = (X1[i].X2[j] + c)^p
        """
        K = np.zeros((X1.shape[0], X2.shape[0]))
        K = (np.dot(X1,np.transpose(X2))+self.c)**self.p
        return K

class SVM(object):

    def __init__(self, kernel, beta=0.001):
        self.kernel = kernel
        self.X = None  # training features
        self.y = None  # training labels
        self.intercept = None
        self.alphas = None
        self.beta = beta  # ridge regularization coefficient
        self.num_classes = 0    # number of classes

    def fit(self, X, y):
        """
        Transform y to (-1,1) and use self.kernel to compute K
        Solve for alphas and compute the intercept using the provided expression
        Keep track of X and y since you'll need them for the prediction
        @param X: data points of shape (num_samples, num_features)
        @param y: (0,1) labels of shape (num_samples,)
        @return: self
        """
        if np.max(y) == 1:
            # binary classification
            self.num_classes = 2
            self.X = X
            self.y = 2 * y - 1  # transforms 0,1 to -1, 1
            K = self.kernel.compute_kernel(self.X,self.X)
            self.alphas = quadratic_solver(K, self.y, self.beta)
            self.intercept = np.mean((self.y - np.sum(self.alphas * self.y * K, axis=1)))
        else:
            # multi-class classification
            # For this assignment, we will use One-vs-Rest strategy
            num_classes = len(list(set(y)))
            self.num_classes = num_classes

            y_one_hot = np.eye(num_classes)[y]
            self.X = X
            self.y = y_one_hot
            K = self.kernel.compute_kernel(self.X, self.X)
            alphas = np.zeros((y.shape[0], num_classes))
            intercept = np.zeros((1, num_classes))
            for i in range(num_classes):
                temp_y = 2 * self.y[:, i] - 1   # transforms 0,1 to -1,1
                alphas[:, i] = quadratic_solver(K, temp_y, self.beta)
                intercept[0, i] = np.mean((temp_y - np.sum(alphas[:, i] * temp_y * K, axis=1)))
                temp = 1
            self.alphas = alphas
            self.intercept = intercept
        return self

    def predict(self, X):
        """
        Predict the labels of points in X
        @param X: samples array of shape (num_samples, num_features)
        @return: predicted 0-1 labels of shape (m,)
        """
        predicted_labels = np.zeros((X.shape[0],))
        if self.num_classes == 2:
            # binary classification
            K = self.kernel.compute_kernel(X, self.X)
            predicted_labels = np.sum(self.y * K * self.alphas, axis=1) + self.intercept
            # sign and convert [-1,1] to [0,1]
            predicted_labels = (np.sign(predicted_labels) + 1) // 2
        else:
            # multi-class classification
            # For this assignment, we will use One-vs-Rest strategy
            predicted_labels_one_hot = np.zeros((X.shape[0], self.num_classes))
            K = self.kernel.compute_kernel(X, self.X)
            for i in range(self.num_classes):
                temp_y = 2 * self.y[:, i] - 1  # transforms 0,1 to -1,1
                predicted_labels_one_hot[:, i] = np.sum(temp_y * K * self.alphas[:, i], axis=1) + self.intercept[0, i]
            predicted_labels = np.argmax(predicted_labels_one_hot, axis=1)
        return predicted_labels

def plot_predict_gt_image(y_true, y_pred_image, filename):
    lines = y_true.shape[0]
    columns = y_true.shape[1]

    # result only for ground reference
    y_true = y_true.reshape(lines * columns, 1)
    y_rgb_pred = cmap_Pavia_U[y_true, :]
    y_image_pred = y_rgb_pred.reshape(lines, columns, 3).astype(np.uint8)
    imageio.imwrite(filename + '_gt_img.png', y_image_pred)

    # result for the whole image
    y_rgb_pred = cmap_Pavia_U[y_pred_image, :]
    y_image_pred = y_rgb_pred.reshape(lines, columns, 3).astype(np.uint8)
    imageio.imwrite(filename + '_predicted_whole_img.png', y_image_pred)


def train_svm(data_file, gt_file, b_pca):
    X, X_img, Y, gt, img = load_matfile(data_file, gt_file, b_pca=b_pca)
    labels = list(set(Y))

    num_classes = 9 # for PU data
    training_num_per_class = 0.2    # 20% for training, 80% for testing

    #################################################
    var_seed = 1
    train_idx, test_idx = training_testing_split_idx(X, Y, num_classes, training_num_per_class, var_seed)
    X_train = X[train_idx].astype('float32')
    y_train = Y[train_idx]

    X_test = X[test_idx].astype('float32')
    y_test = Y[test_idx]

    std = np.std(X_train)
    mean = np.mean(X_train)
    X_train = (X_train - mean) / std

    std = np.std(X_test)
    mean = np.mean(X_test)
    X_test = (X_test - mean) / std

    # svm
    gamma = 1
    svm_classifier = SVM(RadialKernel(gamma))

    svm_classifier.fit(X_train,y_train)
    y_pred = svm_classifier.predict(X_test)

    acc = np.sum(y_pred == y_test) * 1.0 / len(y_test)
    print('acc = ', acc)

    conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
    print("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print(conf_mat)

    kappa = cohen_kappa_score(y_test, y_pred)
    print('Kappa coefficient= ', kappa)

    cla_report = classification_report(y_test, y_pred, labels=labels)
    print(cla_report)

    std = np.std(X_img)
    mean = np.mean(X_img)
    X_img = (X_img - mean) / std
    # process the whole image line-by-line
    nRows = gt.shape[0]
    nCols = gt.shape[1]
    y_pred_image = np.zeros((nRows * nCols, 1))
    for i in range(nRows):
        temp_pred = np.asarray(svm_classifier.predict(X_img[nCols * i: nCols * (i + 1), :])) + 1
        y_pred_image[nCols * i: nCols * (i + 1), 0] = temp_pred
    y_pred_image = np.reshape(y_pred_image, gt.shape)
    y_pred_image = y_pred_image.astype(int)
    plot_predict_gt_image(gt, y_pred_image,
                          filename=r'C:\pingzhang\csci575_final_project_ping\results\with_pca\svm\SVM_PaviaU_kernel_rbf')

    #################################################

    # Save results for SVM.
    f = open(
        r'C:\pingzhang\csci575_final_project_ping\results\with_pca\svm\classify_pavia_U_SVM_results_kernel_rbf.dump',
        'wb')
    pickle.dump(
        (acc, conf_mat, kappa, cla_report), f)
    f.close()

    return acc


def training_testing_split_idx(X, Y, num_classes, percentage_or_num, rep_num):
    ## function description:

    train_idx = []
    train_idx = np.asarray(train_idx).astype('uint32')
    test_idx = []
    test_idx = np.asarray(test_idx).astype('uint32')
    for i in range(num_classes):
        idx = np.where(Y == i)
        idx = np.asarray(idx, dtype='uint').reshape(-1)
        seeds = Random()
        seeds.seed(rep_num)
        seeds.shuffle(idx)

        if percentage_or_num > 1:  # specific number for each class
            train_idx = np.append(train_idx, idx[:percentage_or_num])
            test_idx = np.append(test_idx, idx[percentage_or_num:])
        else:  # specific percentage for each class
            train_idx = np.append(train_idx, idx[:int(percentage_or_num * idx.size)])
            test_idx = np.append(test_idx, idx[int(percentage_or_num * idx.size):])

    return train_idx, test_idx


def load_matfile(data_file, gt_file, b_pca):
    f = scipy.io.loadmat(data_file)
    img = f['paviaU']
    nRows = img.shape[0]
    nCols = img.shape[1]
    nBands = img.shape[2]

    num_pca = 5  # 0.9936695281658365

    f = scipy.io.loadmat(gt_file)
    gt = f['paviaU_gt']
    num_gt_pixel = np.sum(gt != 0)  # 0 is NODATA

    X = np.zeros((num_gt_pixel, nBands))
    X_img = np.zeros((nRows*nCols, nBands))
    Y = np.zeros((num_gt_pixel, 1))

    cnt_gt = 0
    cnt_img = 0
    for i in range(nRows):
        for j in range(nCols):
            X_img[cnt_img, :] = img[i, j, :]
            cnt_img += 1
            if gt[i, j] != 0:
                X[cnt_gt, :] = img[i, j, :]
                Y[cnt_gt] = gt[i, j]
                cnt_gt += 1

    Y = Y.astype(int)
    Y = np.squeeze(Y) - 1

    if b_pca == True:
        # use PCA
        X_pca = np.zeros((num_gt_pixel, num_pca))

        pca = PCA(n_components=num_pca)
        X_img_pca = pca.fit_transform(X_img)
        cnt_gt = 0
        cnt_img = 0
        for i in range(nRows):
            for j in range(nCols):
                if gt[i, j] != 0:
                    X_pca[cnt_gt, :] = X_img_pca[cnt_img, :]
                    # Y[cnt_gt] = gt[i, j]
                    cnt_gt += 1
                cnt_img += 1
        return X_pca, X_img_pca, Y, gt, img

    else:
        # use the original spectral features
        return X, X_img, Y, gt, img


if __name__ == '__main__':
    # Parameter Initialization
    cmap_Pavia_U = np.asarray([[0, 0, 0],
                               [255, 255, 255],  # Asphalt
                               [0, 255, 0],  # Meadows
                               [160, 82, 45],  # Gravel
                               [0, 139, 0],  # Trees
                               [255, 127, 80],  # Painted Metal Sheets
                               [255, 255, 0],  # Bare Soil
                               [255, 0, 0],  # Bitumen
                               [238, 0, 238],  # Self-Blocking Bricks
                               [160, 32, 240]],  # Shadows
                              dtype='int32')

    data_file = r'C:\pingzhang\csci575_final_project_ping\data\PaviaU.mat'
    gt_file = r'C:\pingzhang\csci575_final_project_ping\data\PaviaU_gt.mat'
    b_pca = True
    testing_OA = train_svm(data_file, gt_file, b_pca=b_pca)
