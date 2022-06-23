import glob
import os
import unittest

import cv2
import numpy as np

from utils import read_raw_file

nrows, ncols = 256, 1024


class DualMainTestCase(unittest.TestCase):
    def test_dual_main(self):
        test_img_dirs = '/Volumes/LENOVO_USB_HDD/zhouchao/616_cut/*.raw'
        selected_bands = None
        img_fifo_path = "/tmp/dkimg.fifo"
        mask_fifo_path = "/tmp/dkmask.fifo"

        total_len = nrows * ncols
        spectral_files = glob.glob(test_img_dirs)
        print("reading raw files ...")
        raw_files = [read_raw_file(file, selected_bands=selected_bands) for file in spectral_files]
        print("reading file success!")
        if not os.access(img_fifo_path, os.F_OK):
            os.mkfifo(img_fifo_path, 0o777)
        if not os.access(mask_fifo_path, os.F_OK):
            os.mkfifo(mask_fifo_path, 0o777)
        data = b''
        for raw_file in raw_files:
            if raw_file.shape[0] > nrows:
                raw_file = raw_file[:nrows, ...]
            # 写出
            print(f"send {raw_file.shape}")
            fd_img = os.open(img_fifo_path, os.O_WRONLY)
            os.write(fd_img, raw_file.tobytes())
            os.close(fd_img)
            # 等待
            fd_mask = os.open(mask_fifo_path, os.O_RDONLY)
            while len(data) < total_len:
                data += os.read(fd_mask, total_len)
            if len(data) > total_len:
                data_total = data[:total_len]
                data = data[total_len:]
            else:
                data_total = data
                data = b''
            os.close(fd_mask)
            mask = np.frombuffer(data_total, dtype=np.uint8).reshape((-1, ncols))

            # 显示
            rgb_img = np.asarray(raw_file[..., [0, 2, 3]] * 255, dtype=np.uint8)
            mask_color = np.zeros_like(rgb_img)
            mask_color[mask > 0] = (0, 0, 255)
            combine = cv2.addWeighted(rgb_img, 1, mask_color, 0.5, 0)
            cv2.imshow("img", combine)
            cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
