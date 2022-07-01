import os
import pickle
import time

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nrows, ncols, nbands = 256, 1024, 4


def feature(x):
    x = x.reshape((x.shape[0], -1))
    return x


def train_rf_and_report(train_x, train_y, test_x, test_y,
                        tree_num, save_path=None):
    rfc = RandomForestClassifier(n_estimators=tree_num, random_state=42, class_weight={0: 10, 1: 10})
    rfc = rfc.fit(train_x, train_y)
    t1 = time.time()
    y_pred = rfc.predict(test_x)
    y_pred_binary = np.ones_like(y_pred)
    y_pred_binary[(y_pred == 0) | (y_pred == 1)] = 0
    y_pred_binary[(y_pred > 1)] = 2
    test_y_binary = np.ones_like(test_y)
    test_y_binary[(test_y == 0) | (test_y == 1)] = 0
    test_y_binary[(test_y > 1)] = 2
    print("预测时间：", time.time() - t1)
    print("RFC训练模型评分：" + str(accuracy_score(train_y, rfc.predict(train_x))))
    print("RFC待测模型评分：" + str(accuracy_score(test_y, rfc.predict(test_x))))
    print('RFC预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('RFC分类报告\n' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('RFC混淆矩阵：\n' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少
    print('rfc分类报告：\n' + str(classification_report(test_y_binary, y_pred_binary)))  # 生成一个小报告呀
    print('rfc混淆矩阵：\n' + str(confusion_matrix(test_y_binary, y_pred_binary)))  # 这个也是，生成的矩阵的意思是有多少
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(rfc, f)
    return rfc


def train_t_and_report(train_x, train_y, test_x, test_y, save_path=None):
    rfc = DecisionTreeClassifier(random_state=42, class_weight={0: 10, 1: 10})
    rfc = rfc.fit(train_x, train_y)
    t1 = time.time()
    y_pred = rfc.predict(test_x)
    y_pred_binary = np.ones_like(y_pred)
    y_pred_binary[(y_pred == 0) | (y_pred == 1)] = 0
    y_pred_binary[(y_pred > 1)] = 2
    test_y_binary = np.ones_like(test_y)
    test_y_binary[(test_y == 0) | (test_y == 1)] = 0
    test_y_binary[(test_y > 1)] = 2
    print("预测时间：", time.time() - t1)
    print("训练集acc：" + str(accuracy_score(train_y, rfc.predict(train_x))))
    print("测试集acc：" + str(accuracy_score(test_y, rfc.predict(test_x))))
    print('-'*50)
    print('测试集报告\n' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('混淆矩阵：\n' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少
    print('二分类报告：\n' + str(classification_report(test_y_binary, y_pred_binary)))  # 生成一个小报告呀
    print('二混淆矩阵：\n' + str(confusion_matrix(test_y_binary, y_pred_binary)))  # 这个也是，生成的矩阵的意思是有多少
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(rfc, f)
    return rfc


def evaluation_and_report(model, test_x, test_y):
    t1 = time.time()
    y_pred = model.predict(test_x)
    y_pred_binary = np.ones_like(y_pred)
    y_pred_binary[(y_pred == 0) | (y_pred == 1)] = 0
    y_pred_binary[(y_pred > 1)] = 2
    test_y_binary = np.ones_like(test_y)
    test_y_binary[(test_y == 0) | (test_y == 1)] = 0
    test_y_binary[(test_y > 1)] = 2
    print("预测时间：", time.time() - t1)
    print("RFC待测模型 accuracy：" + str(accuracy_score(test_y, model.predict(test_x))))
    print('RFC预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('RFC分类报告\n' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('RFC混淆矩阵：\n' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少
    print('rfc分类报告：\n' + str(classification_report(test_y_binary, y_pred_binary)))  # 生成一个小报告呀
    print('rfc混淆矩阵：\n' + str(confusion_matrix(test_y_binary, y_pred_binary)))  # 这个也是，生成的矩阵的意思是有多少


def train_pca_rf(train_x, train_y, test_x, test_y, n_comp,
                 tree_num, save_path=None):
    rfc = RandomForestClassifier(n_estimators=tree_num, random_state=42, class_weight={0: 100, 1: 100})
    pca = PCA(n_components=n_comp)
    pca = pca.fit(train_x)
    pca_train_x = pca.transform(train_x)
    rfc = rfc.fit(pca_train_x, train_y)
    t1 = time.time()
    pca_test_x = pca.transform(test_x)
    y_pred = rfc.predict(pca_test_x)
    y_pred_binary = np.ones_like(y_pred)
    y_pred_binary[(y_pred == 0) | (y_pred == 1)] = 0
    y_pred_binary[(y_pred == 2) | (y_pred == 3) | (y_pred == 4)] = 2
    test_y_binary = np.ones_like(test_y)
    test_y_binary[(test_y == 0) | (test_y == 1)] = 0
    test_y_binary[(test_y == 2) | (test_y == 3) | (test_y == 4)] = 2
    print("预测时间：", time.time() - t1)
    print("RFC训练模型评分：" + str(accuracy_score(train_y, rfc.predict(pca_train_x))))
    print("RFC待测模型评分：" + str(accuracy_score(test_y, rfc.predict(pca_test_x))))
    print('RFC预测结果：' + str(y_pred))
    print('---------------------------------------------------------------------------------------------------')
    print('RFC分类报告：\n' + str(classification_report(test_y, y_pred)))  # 生成一个小报告呀
    print('RFC混淆矩阵：\n' + str(confusion_matrix(test_y, y_pred)))  # 这个也是，生成的矩阵的意思是有多少
    print('rfc分类报告：\n' + str(classification_report(test_y_binary, y_pred_binary)))  # 生成一个小报告呀
    print('rfc混淆矩阵：\n' + str(confusion_matrix(test_y_binary, y_pred_binary)))  # 这个也是，生成的矩阵的意思是有多少
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump((pca, rfc), f)
    return pca, rfc


def split_x(data: np.ndarray, blk_sz: int) -> list:
    """
    Split the data into slices for classification.将数据划分为多个像素块,便于后续识别.

    ;param data: image data, shape (num_rows x ncols x num_channels)
    ;param blk_sz: block size
    ;param sensitivity: 最少有多少个杂物点能够被认为是杂物
    ;return data_x, data_y: sliced data x (block_num x num_charnnels x blk_sz x blk_sz)
    """
    x_list = []
    for i in range(0, nrows // blk_sz):
        for j in range(0, ncols // blk_sz):
            block_data = data[i * blk_sz: (i + 1) * blk_sz, j * blk_sz: (j + 1) * blk_sz, ...]
            x_list.append(block_data)
    return x_list


class SpecDetector(object):
    def __init__(self, model_path, blk_sz=8, channel_num=nbands):
        self.blk_sz, self.channel_num = blk_sz, channel_num
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                self.clf = pickle.load(model_file)
        else:
            raise FileNotFoundError("Model File not found")

    def predict(self, data, rigor_rate=100):
        blocks = split_x(data, blk_sz=self.blk_sz)
        blocks = np.array(blocks)
        features = feature(np.array(blocks))
        # print("Spec Detector", rigor_rate)
        y_pred = self.clf.predict_proba(features)
        y_pred, y_prob = np.argmax(y_pred, axis=1), np.max(y_pred, axis=1)
        y_pred_binary = np.zeros_like(y_pred)
        # classes merge
        y_pred_binary[((y_pred == 2) | (y_pred >= 3)) & (y_prob > (100 - rigor_rate) / 100.0)] = 1
        # transform to mask
        mask = self.mask_transform(y_pred_binary, (ncols, nrows))
        return mask

    def mask_transform(self, result, dst_size):
        mask_size = nrows // self.blk_sz, ncols // self.blk_sz
        mask = np.zeros(mask_size, dtype=np.uint8)
        for idx, r in enumerate(result):
            row, col = idx // mask_size[1], idx % mask_size[1]
            mask[row, col] = r
        mask = mask.repeat(self.blk_sz, axis=0).repeat(self.blk_sz, axis=1)
        return mask


class PixelWisedDetector(object):
    def __init__(self, model_path, blk_sz=1, channel_num=nbands):
        self.blk_sz, self.channel_num = blk_sz, channel_num
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                self.clf = pickle.load(model_file)
        else:
            raise FileNotFoundError("Model File not found")

    def predict(self, data, rigor_rate=70):
        features = data.reshape((-1, self.channel_num))
        y_pred = self.clf.predict(features, rigor_rate)
        y_pred_binary = np.ones_like(y_pred, dtype=np.uint8)
        # print("pixel detector", rigor_rate)
        # classes merge
        y_pred_binary[(y_pred == 0) | (y_pred == 1) | (y_pred == 3)] = 0
        # transform to mask
        mask = self.mask_transform(y_pred_binary)
        return mask

    def mask_transform(self, result):
        mask_size = (nrows, ncols)
        mask = result.reshape(mask_size)
        return mask


class PcaSpecDetector(object):
    def __init__(self, model_path, pca_path, blk_sz=8, channel_num=nbands):
        self.blk_sz, self.channel_num = blk_sz, channel_num
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                self.clf = pickle.load(model_file)
        else:
            raise FileNotFoundError("Model File not found")
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as pca_file:
                self.pca = pickle.load(pca_file)
        else:
            raise FileNotFoundError("Pca File not found")

    def predict(self, data):
        blocks = split_x(data, blk_sz=self.blk_sz)
        blocks = np.array(blocks)
        features = feature(np.array(blocks))
        y_pred = self.clf.predict(features)
        y_pred_binary = np.ones_like(y_pred)
        # classes merge
        y_pred_binary[(y_pred == 0) | (y_pred == 1) | (y_pred == 3)] = 0
        # transform to mask
        mask = self.mask_transform(y_pred_binary, (ncols, nrows))
        return mask

    def mask_transform(self, result, dst_size):
        mask_size = nrows // self.blk_sz, ncols // self.blk_sz
        mask = np.zeros(mask_size, dtype=np.uint8)
        for idx, r in enumerate(result):
            row, col = idx // mask_size[1], idx % mask_size[1]
            mask[row, col] = r
        mask = mask.repeat(self.blk_sz, axis=0).repeat(self.blk_sz, axis=1)
        return mask
