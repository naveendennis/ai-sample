import unittest
import numpy as np
import neural_network.train_evaluate_amazon_nn as azn


class AmazonNN(unittest.TestCase):
    def test_get_label_indices(self):
        indices = np.array([1,2,3,3,4,5])
        self.assertEqual(azn.get_label_indices(indices), np.array([[0],[1],[2,3],[4],[5]]))
