import os
import time

import cv2
import numpy as np
from models import SpecDetector
from root_dir import ROOT_DIR

nrows, ncols, nbands = 256, 1024, 4
img_fifo_path = "/tmp/dkimg.fifo"
mask_fifo_path = "/tmp/dkmask.fifo"
selected_model = "rf_8x8_c4_185_sen32_4.model"


def main():
    model_path = os.path.join(ROOT_DIR, "models", selected_model)
    detector = SpecDetector(model_path, blk_sz=8, channel_num=4)
    t1 = time.time()
    _ = detector.predict(np.zeros((nrows, ncols, nbands)))
    t2 = time.time()
    _ = detector.predict(np.zeros((nrows, ncols, nbands)))
    t3 = time.time()
    _ = detector.predict(np.ones((nrows, ncols, nbands)))
    t4 = time.time()
    print(f"First time test {(t2 -t1)*1000:.2f}ms\n"
          f"Second time test {(t3 - t2)*1000:.2f}ms\n"
          f"Third time test {(t4 - t3)*1000:.2f}ms\n")
    total_len = nrows * ncols * nbands * 4

    if not os.access(img_fifo_path, os.F_OK):
        os.mkfifo(img_fifo_path, 0o777)
    if not os.access(mask_fifo_path, os.F_OK):
        os.mkfifo(mask_fifo_path, 0o777)
    data = b''
    print("Start the serveice.")
    print("="*20)
    while True:
        # 读取
        fd_img = os.open(img_fifo_path, os.O_RDONLY)
        while len(data) < total_len:
            data += os.read(fd_img, total_len)
        if len(data) > total_len:
            data_total = data[:total_len]
            data = data[total_len:]
        else:
            data_total = data
            data = b''

        os.close(fd_img)
        # 识别
        img = np.frombuffer(data_total, dtype=np.float32).reshape((nrows, nbands, -1)).transpose(0, 2, 1)
        t1 = time.time()
        mask = detector.predict(img)
        print(f"time spent {(time.time() - t1)*1000:.2f}ms")
        # 写出
        fd_mask = os.open(mask_fifo_path, os.O_WRONLY)
        os.write(fd_mask, mask.tobytes())
        os.close(fd_mask)


if __name__ == '__main__':
    main()
