import os
import numpy as np
from models import SpecDetector
from root_dir import ROOT_DIR

nrows, ncols, nbands = 600, 1024, 4
img_fifo_path = "/tmp/dkimg.fifo"
mask_fifo_path = "/tmp/dkmask.fifo"
selected_model = "rf_8x8_c4_400_13.model"

def main():
    model_path = os.path.join(ROOT_DIR, "models", selected_model)
    detector = SpecDetector(model_path, blk_sz=8, channel_num=4)
    _ = detector.predict(np.ones((600, 1024, 4)))
    total_len = nrows * ncols * nbands * 4
    if not os.access(img_fifo_path, os.F_OK):
        os.mkfifo(img_fifo_path, 0o777)
    if not os.access(mask_fifo_path, os.F_OK):
        os.mkfifo(mask_fifo_path, 0o777)

    fd_img = os.open(img_fifo_path, os.O_RDONLY)
    print("connect to fifo")

    while True:
        data = os.read(fd_img, total_len)
        print("get img")
        img = np.frombuffer(data, dtype=np.float32).reshape((nrows, nbands, -1)).transpose(0, 2, 1)
        mask = detector.predict(img)
        fd_mask = os.open(mask_fifo_path, os.O_WRONLY)
        os.write(fd_mask, mask.tobytes())
        os.close(fd_mask)


if __name__ == '__main__':
    main()
