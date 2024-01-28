import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numpy.random import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from random import *
import pickle

import imageio

from sklearn.decomposition import PCA


def most_common(lst):
    return max(set(lst), key=lst.count)
def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KNN:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        # create similarity matrix
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

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


def train_knn(data_file, gt_file, b_pca):
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
    X_test = (X_test - mean) / std

    ### KNN classification
    parameter_k = 5
    knn = KNN(k=parameter_k)
    knn.fit(X_train, y_train)
    ###

    y_pred = knn.predict(X_test)

    acc = np.sum(y_pred == y_test) * 1.0 / len(y_test)
    print('acc= ', acc)

    conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
    print("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print(conf_mat)

    kappa = cohen_kappa_score(y_test, y_pred)
    print('Kappa coefficient= ', kappa)

    cla_report = classification_report(y_test, y_pred, labels=labels)
    print(cla_report)

    X_img = (X_img - mean) / std
    y_pred_image = np.asarray(knn.predict(X_img)) + 1
    y_pred_image = np.reshape(y_pred_image, gt.shape)
    plot_predict_gt_image(gt, y_pred_image, filename=r'C:\pingzhang\csci575_final_project_ping\results\with_pca\knn\knn_PaviaU')
    #################################################

    # Save results for knn
    f = open(
        r'C:\pingzhang\csci575_final_project_ping\results\with_pca\knn\classify_pavia_U_knn_results.dump', 'wb')
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

    num_pca = 5 # 0.9936695281658365

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
    testing_OA = train_knn(data_file, gt_file, b_pca=b_pca)