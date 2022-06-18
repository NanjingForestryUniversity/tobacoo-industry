import cv2
import numpy as np

import glob
import os
import time

import matplotlib.pyplot as plt

from models import SpecDetector


def trans_color(pixel: np.ndarray, color_dict: dict = None) -> int:
    """
    将label转为类别

    :param pixel: 一个 n x n 的像素块
    :param color_dict: 用于转化的字典 {(0, 0, 255): 1, ....} 色彩采用bgr
    :return:类别白噢好
    """
    # 0 表示的是背景， 1表示的是烟梗，剩下的都是杂质
    if color_dict is None:
        color_dict = {(0, 0, 255): 1, (255, 255, 255): 0, (0, 255, 0): 2, (255, 255, 0): 3, (0, 255, 255): 4}
    if (pixel[0], pixel[1], pixel[2]) in color_dict.keys():
        return color_dict[(pixel[0], pixel[1], pixel[2])]
    else:
        return -1


def determine_class(pixel_blk: np.ndarray, sensitivity=8) -> int:
    """
    决定像素块的类别

    :param pixel_blk: 像素块
    :param sensitivity: 敏感度
    :return:
    """
    defect_dict = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    color_numbers = {cls: pixel_blk.shape[0] ** 2 - np.count_nonzero(pixel_blk - cls)
                     for cls in defect_dict.keys()}
    grant_cls = {0: 0, 1: 0}
    for cls, num in color_numbers.items():
        grant_cls[defect_dict[cls]] += num
    if grant_cls[1] >= sensitivity:
        color_numbers = {cls: color_numbers[cls] for cls in [2, 3, 4, 5, 6]}
        return max(color_numbers, key=color_numbers.get)
    else:
        if color_numbers[1] >= sensitivity:
            return 1
        return 0


def split_xy(data: np.ndarray, labeled_img: np.ndarray, blk_sz: int, sensitivity: int = 12,
             color_dict=None, add_background=True) -> tuple:
    """
    Split the data into slices for classification.将数据划分为多个像素块,便于后续识别.

    ;param data: image data, shape (num_rows x 1024 x num_channels)
    ;param labeled_img: RGB labeled img with respect to the image!
                        make sure that the defect is (255, 0, 0) and background is (255, 255, 255)
    ;param blk_sz: block size
    ;param sensitivity: 最少有多少个杂物点能够被认为是杂物
    ;return data_x, data_y: sliced data x (block_num x num_charnnels x blk_sz x blk_sz)
                            data y (block_num, ) 1 是杂质， 0是无杂质
    """
    assert (data.shape[0] == labeled_img.shape[0]) and (data.shape[1] == labeled_img.shape[1])
    color_dict = {(0, 0, 255): 1, (255, 255, 255): 0, (0, 255, 0): 2, (255, 255, 0): 3, (0, 255, 255): 4}\
                if color_dict is None else color_dict
    class_img = np.zeros((labeled_img.shape[0], labeled_img.shape[1]), dtype=int)
    for color, class_idx in color_dict.items():
        truth_map = np.all(labeled_img == color, axis=2)
        class_img[truth_map] = class_idx
    x_list, y_list = [], []
    for i in range(0, 600 // blk_sz):
        for j in range(0, 1024 // blk_sz):
            block_data = data[i * blk_sz: (i + 1) * blk_sz, j * blk_sz: (j + 1) * blk_sz, ...]
            block_label = class_img[i * blk_sz: (i + 1) * blk_sz, j * blk_sz: (j + 1) * blk_sz, ...]
            block_label = determine_class(block_label, sensitivity=sensitivity)
            if add_background:
                y_list.append(block_label)
                x_list.append(block_data)
            else:
                if block_label != 0:
                    y_list.append(block_label)
                    x_list.append(block_data)
    return x_list, y_list


def split_x(data: np.ndarray, blk_sz: int) -> list:
    """
    Split the data into slices for classification.将数据划分为多个像素块,便于后续识别.

    ;param data: image data, shape (num_rows x 1024 x num_channels)
    ;param blk_sz: block size
    ;param sensitivity: 最少有多少个杂物点能够被认为是杂物
    ;return data_x, data_y: sliced data x (block_num x num_charnnels x blk_sz x blk_sz)
    """
    x_list = []
    for i in range(0, 600 // blk_sz):
        for j in range(0, 1024 // blk_sz):
            block_data = data[i * blk_sz: (i + 1) * blk_sz, j * blk_sz: (j + 1) * blk_sz, ...]
            x_list.append(block_data)
    return x_list


def visualization_evaluation(detector, data_path, selected_bands=None):
    selected_bands = [76, 146, 216, 367, 383, 406] if selected_bands is None else selected_bands
    nrows, ncols = 600, 1024
    image_paths = glob.glob(os.path.join(data_path, "calibrated*.raw"))
    for idx, image_path in enumerate(image_paths):
        with open(image_path, 'rb') as f:
            data = f.read()
        img = np.frombuffer(data, dtype=np.float32).reshape((nrows, -1, ncols)).transpose(0, 2, 1)
        nbands = img.shape[2]
        t1 = time.time()
        mask = detector.predict(img[..., selected_bands] if nbands == 448 else img)
        time_spent = time.time() - t1
        if nbands == 448:
            rgb_img = np.asarray(img[..., [372, 241, 169]] * 255, dtype=np.uint8)
        else:
            rgb_img = np.asarray(img[..., [0, 1, 2]] * 255, dtype=np.uint8)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(rgb_img)
        axs[1].imshow(mask)
        fig.suptitle(f"time spent {time_spent*1000:.2f} ms" + f"\n{image_path}")
        plt.savefig(f"./dataset/{idx}.png", dpi=300)
        plt.show()


def visualization_y(y_list, k_size):
    mask = np.zeros((600//k_size, 1024//k_size), dtype=np.uint8)
    for idx, r in enumerate(y_list):
        row, col = idx // (1024 // k_size), idx % (1024 // k_size)
        mask[row, col] = r
    fig, axs = plt.subplots()
    axs.imshow(mask)
    plt.show()


def read_raw_file(file_name, selected_bands=None):
    with open(file_name, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32).reshape((600, -1, 1024)).transpose(0, 2, 1)
    if selected_bands is not None:
        data = data[..., selected_bands]
    return data


def read_black_and_white_file(file_name):
    with open(file_name, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32).reshape((1, 448, 1024)).transpose(0, 2, 1)
    return data


def label2pic(label, color_dict):
    pic = np.zeros((label.shape[0], label.shape[1], 3))
    for color, cls in color_dict.items():
        pic[label == cls] = color
    return pic


def generate_tobacco_label(data, model_file, blk_sz, selected_bands):
    model = SpecDetector(model_path=model_file, blk_sz=blk_sz, channel_num=len(selected_bands))
    y_label = model.predict(data)
    x_list, y_list = [], []
    for i in range(0, 600 // blk_sz):
        for j in range(0, 1024 // blk_sz):
            if np.sum(np.sum(y_label[i * blk_sz: (i + 1) * blk_sz, j * blk_sz: (j + 1) * blk_sz, ...])) \
                    > 0:
                block_data = data[i * blk_sz: (i + 1) * blk_sz, j * blk_sz: (j + 1) * blk_sz, ...]
                x_list.append(block_data)
                y_list.append(1)
    return x_list, y_list


def generate_impurity_label(data, light_threshold, color_dict, split_line=0,  target_class_right=None,
                            target_class_left=None,):
    y_label = np.zeros((data.shape[0], data.shape[1]))
    for i in range(0, 600):
        for j in range(0, 1024):
            if np.sum(np.sum(data[i, j])) >= light_threshold:
                if j > split_line:
                    y_label[i, j] = target_class_right
                else:
                    y_label[i, j] = target_class_left
    pic = label2pic(y_label, color_dict=color_dict)
    fig, axs = plt.subplots(2, 1)
    axs[0].matshow(y_label)
    axs[1].matshow(data[..., 0])
    plt.show()
    return pic
