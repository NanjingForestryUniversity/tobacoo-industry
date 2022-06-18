import unittest

import numpy as np

from utils import determine_class, split_xy, split_x


class DatasetTest(unittest.TestCase):
    def test_determine_class(self):
        pixel_block = np.zeros((8, 8), dtype=np.uint8)
        pixel_block[2: 4, 5: 6] = 2
        pixel_block[5: 7, 1: 2] = 1
        pixel_block[4: 6, 1: 6] = 3
        cls = determine_class(pixel_block, sensitivity=8)
        self.assertEqual(cls, 3)
        pixel_block = np.zeros((8, 8), dtype=np.uint8)
        pixel_block[2: 4, 5: 6] = 2
        pixel_block[5: 7, 1: 2] = 1
        pixel_block[4: 6, 1: 6] = 2
        cls = determine_class(pixel_block, sensitivity=8)
        self.assertEqual(cls, 2)
        pixel_block = np.zeros((8, 8), dtype=np.uint8)
        pixel_block[2: 4, 5: 6] = 1
        pixel_block[5: 7, 1: 2] = 2
        pixel_block[4: 6, 1: 6] = 1
        cls = determine_class(pixel_block, sensitivity=8)
        self.assertEqual(cls, 1)

    def test_split_xy(self):
        x = np.arange(600*1024).reshape((600, 1024))
        y = np.zeros((600, 1024, 3))
        color_dict = {(0, 0, 255): 1, (255, 255, 255): 0, (0, 255, 0): 2, (255, 255, 0): 3, (0, 255, 255): 4}
        trans_color_dict = {v: k for k, v in color_dict.items()}
        # modify the first block
        y[2: 4, 5: 6] = trans_color_dict[2]
        y[5: 7, 1: 2] = trans_color_dict[1]
        y[4: 6, 1: 6] = trans_color_dict[3]
        # modify the last block
        y[-4: -2, -6: -5] = trans_color_dict[1]
        y[-7: -5, -2: -1] = trans_color_dict[2]
        y[-6: -4, -6: -1] = trans_color_dict[1]
        # modify the middle block
        y[64+2: 64+4, 64+5: 64+6] = trans_color_dict[2]
        y[64+5: 64+7, 64+1: 64+2] = trans_color_dict[1]
        y[64+4: 64+6, 64+1: 64+6] = trans_color_dict[2]
        x_list, y_list = split_xy(x, y, blk_sz=8, sensitivity=8)
        first_block = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
        [1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031],
        [2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055],
        [3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079],
        [4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103],
        [5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127],
        [6144, 6145, 6146, 6147, 6148, 6149, 6150, 6151],
        [7168, 7169, 7170, 7171, 7172, 7173, 7174, 7175],])
        sum_value = np.sum(np.sum(x_list[0]-first_block))
        self.assertEqual(sum_value, 0)
        self.assertEqual(y_list[0], 3)
        last_block = np.array(
        [[607224, 607225, 607226, 607227, 607228, 607229, 607230, 607231],
        [608248, 608249, 608250, 608251, 608252, 608253, 608254, 608255],
        [609272, 609273, 609274, 609275, 609276, 609277, 609278, 609279],
        [610296, 610297, 610298, 610299, 610300, 610301, 610302, 610303],
        [611320, 611321, 611322, 611323, 611324, 611325, 611326, 611327],
        [612344, 612345, 612346, 612347, 612348, 612349, 612350, 612351],
        [613368, 613369, 613370, 613371, 613372, 613373, 613374, 613375],
        [614392, 614393, 614394, 614395, 614396, 614397, 614398, 614399],])
        sum_value = np.sum(np.sum(x_list[-1]-last_block))
        self.assertEqual(sum_value, 0)
        self.assertEqual(y_list[-1], 1)
        middle_block = np.array([[65600, 65601, 65602, 65603, 65604, 65605, 65606, 65607],
               [66624, 66625, 66626, 66627, 66628, 66629, 66630, 66631],
               [67648, 67649, 67650, 67651, 67652, 67653, 67654, 67655],
               [68672, 68673, 68674, 68675, 68676, 68677, 68678, 68679],
               [69696, 69697, 69698, 69699, 69700, 69701, 69702, 69703],
               [70720, 70721, 70722, 70723, 70724, 70725, 70726, 70727],
               [71744, 71745, 71746, 71747, 71748, 71749, 71750, 71751],
               [72768, 72769, 72770, 72771, 72772, 72773, 72774, 72775]])
        sum_value = np.sum(np.sum(x_list[1032]-middle_block))
        self.assertEqual(sum_value, 0)
        self.assertEqual(y_list[1032], 2)

    def test_split_x(self):
        x = np.arange(600 * 1024).reshape((600, 1024))
        x_list = split_x(x, blk_sz=8)
        first_block = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                [1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031],
                                [2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055],
                                [3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079],
                                [4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103],
                                [5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127],
                                [6144, 6145, 6146, 6147, 6148, 6149, 6150, 6151],
                                [7168, 7169, 7170, 7171, 7172, 7173, 7174, 7175], ])
        sum_value = np.sum(np.sum(x_list[0] - first_block))
        self.assertEqual(sum_value, 0)
        last_block = np.array(
            [[607224, 607225, 607226, 607227, 607228, 607229, 607230, 607231],
             [608248, 608249, 608250, 608251, 608252, 608253, 608254, 608255],
             [609272, 609273, 609274, 609275, 609276, 609277, 609278, 609279],
             [610296, 610297, 610298, 610299, 610300, 610301, 610302, 610303],
             [611320, 611321, 611322, 611323, 611324, 611325, 611326, 611327],
             [612344, 612345, 612346, 612347, 612348, 612349, 612350, 612351],
             [613368, 613369, 613370, 613371, 613372, 613373, 613374, 613375],
             [614392, 614393, 614394, 614395, 614396, 614397, 614398, 614399], ])
        sum_value = np.sum(np.sum(x_list[-1] - last_block))
        self.assertEqual(sum_value, 0)
        middle_block = np.array([[65600, 65601, 65602, 65603, 65604, 65605, 65606, 65607],
                                 [66624, 66625, 66626, 66627, 66628, 66629, 66630, 66631],
                                 [67648, 67649, 67650, 67651, 67652, 67653, 67654, 67655],
                                 [68672, 68673, 68674, 68675, 68676, 68677, 68678, 68679],
                                 [69696, 69697, 69698, 69699, 69700, 69701, 69702, 69703],
                                 [70720, 70721, 70722, 70723, 70724, 70725, 70726, 70727],
                                 [71744, 71745, 71746, 71747, 71748, 71749, 71750, 71751],
                                 [72768, 72769, 72770, 72771, 72772, 72773, 72774, 72775]])
        sum_value = np.sum(np.sum(x_list[1032] - middle_block))
        self.assertEqual(sum_value, 0)


if __name__ == '__main__':
    unittest.main()
