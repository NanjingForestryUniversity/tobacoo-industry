import unittest

import numpy as np

from models import feature


class ModelTestCase(unittest.TestCase):
    def test_feature(self):
        x_list = [np.ones((8, 8, 6)) * i for i in range(9600)]
        features = feature(x_list=x_list)
        self.assertEqual(features[0][0], 0)  # add assertion here
        self.assertEqual(features[0][5], 0)  # add assertion here
        self.assertEqual(features[-1][5], 9599)  # add assertion here
        self.assertEqual(features[-1][0], 9599)  # add assertion here


if __name__ == '__main__':
    unittest.main()
