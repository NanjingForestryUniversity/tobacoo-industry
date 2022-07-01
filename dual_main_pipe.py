import os
import time

import numpy as np
from models import SpecDetector, PixelWisedDetector
from root_dir import ROOT_DIR
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

nrows, ncols, nbands = 256, 1024, 4

img_fifo_path = "/tmp/dkimg.fifo"
mask_fifo_path = "/tmp/dkmask.fifo"
cmd_fifo_path = '/tmp/tobacco_cmd.fifo'

pxl_model_path = "rf_1x1_c4_1_sen1_4.model"
blk_model_path = "rf_8x8_c4_185_sen32_4.model"


def main(pxl_model_path=pxl_model_path, blk_model_path=blk_model_path):
    # make fifos to communicate with the child model processes
    blk_img_pipe_parent, blk_img_img_pipe_child = Pipe()
    blk_msk_pipe_parent, blk_msk_pipe_child = Pipe()
    blk_cmd_pipe_parent, blk_cmd_pipe_child = Pipe()
    blk_process = Process(target=model_process_func,
                          args=(blk_cmd_pipe_child, blk_img_img_pipe_child,
                                blk_msk_pipe_child, "blk", blk_model_path, ))
    pxl_img_pipe_parent, pxl_img_img_pipe_child = Pipe()
    pxl_msk_pipe_parent, pxl_msk_pipe_child = Pipe()
    pxl_cmd_pipe_parent, pxl_cmd_pipe_child = Pipe()
    pxl_process = Process(target=model_process_func,
                          args=(pxl_cmd_pipe_child, pxl_img_img_pipe_child,
                                pxl_cmd_pipe_child, "pxl", blk_model_path, ))

    blk_process.start()
    pxl_process.start()
    total_len = nrows * ncols * nbands * 4
    if not os.access(img_fifo_path, os.F_OK):
        os.mkfifo(img_fifo_path, 0o777)
    if not os.access(mask_fifo_path, os.F_OK):
        os.mkfifo(mask_fifo_path, 0o777)
    data = b''
    while True:
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
        t1 = time.time()
        img = np.frombuffer(data_total, dtype=np.float32).reshape((nrows, nbands, -1)).transpose(0, 2, 1)
        print(f"get img shape {img.shape}")
        pxl_img_queue.put(img)
        blk_img_queue.put(img)
        pxl_msk = pxl_msk_queue.get()
        blk_msk = blk_msk_queue.get()
        mask = pxl_msk & blk_msk
        print(f"predict success get mask shape: {mask.shape}")
        print(f"Time: {time.time() - t1}")
        # 写出
        fd_mask = os.open(mask_fifo_path, os.O_WRONLY)
        os.write(fd_mask, mask.tobytes())
        os.close(fd_mask)


def model_process_func(cmd_pipe: Connection, img_pipe: Connection, msk_pipe: Connection,
                       model_cls: str, model_path=pxl_model_path):
    assert model_cls in ['pxl', 'blk']
    if model_cls == 'pxl':
        model = PixelWisedDetector(os.path.join(ROOT_DIR, "models", model_path),
                                   blk_sz=1, channel_num=4)
    else:
        model = SpecDetector(os.path.join(ROOT_DIR, "models", model_path),
                             blk_sz=8, channel_num=4)
    _ = model.predict(np.ones((nrows, ncols, nbands)))
    rigor_rate = 70
    while True:
        # deal with the cmd if cmd_queue is not empty
        if not cmd_pipe.poll():
            cmd = cmd_pipe.recv()
            if isinstance(cmd, int):
                rigor_rate = cmd
            elif isinstance(cmd, str):
                if cmd == 'stop':
                    break
                else:
                    try:
                        if model_cls == 'pxl':
                            model = PixelWisedDetector(os.path.join(ROOT_DIR, "models", model_path),
                                                       blk_sz=1, channel_num=4)
                        else:
                            model = SpecDetector(os.path.join(ROOT_DIR, "models", model_path),
                                                 blk_sz=8, channel_num=4)
                    except Exception as e:
                        print(f"Load Model Failed! {e}")
        # deal with the img if img_queue is not empty
        if not img_pipe.poll():
            t1 = time.time()
            img = img_pipe.recv()
            t2 = time.time()
            mask = model.predict(img, rigor_rate)
            t3 = time.time()
            msk_pipe.send(mask)
            t4 = time.time()
            print(f"{model_cls} model recv time: {(t2 - t1) * 1000}ms\n"
                  f"{model_cls} model predict time: {(t3 - t2) * 1000}ms\n"
                  f"{model_cls} model send time: {(t4 - t3) * 1000}ms")


if __name__ == '__main__':
    main()
