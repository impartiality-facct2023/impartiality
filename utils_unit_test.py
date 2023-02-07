import unittest
import numpy as np
from utils import transform_votes_powerset
from utils import pick_votes_from_probabilities
import torch


class TestUtils(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_transform_votes_powerset(self):
        votes = np.array([
            [[0, 0, 0, 1], [0, 1, 1, 1]],
            [[0, 0, 0, 1], [1, 1, 1, 0]]
        ])
        votes, map_idx_label = transform_votes_powerset(
            input_votes=votes)
        print('votes: ', votes)
        print('map_idx_label: ', map_idx_label)
        np.testing.assert_equal(votes[0], [0, 1, 0, 0, 0, 0, 0, 1,
                                           0, 0, 0, 0, 0, 0, 0, 0])

    def test_pick_votes_from_probabilities(self):
        probs = torch.tensor([
            [
                [0.2, 0.3, 0.5],
                [0.6, 0.3, 0.5],
                [0.0, 0.0, 0.5],
            ],
            [
                [0., 0.0, 0.0],
                [0.7, 0., 0.],
                [0.1, 0.3, 0.0],
            ]
        ])
        votes = pick_votes_from_probabilities(
            probs=probs, powerset_tau=2, threshold=0.1)
        print('votes: ', votes)
        votes = votes.cpu().numpy()
        np.testing.assert_equal(votes.sum(), 8)

        votes = pick_votes_from_probabilities(
            probs=probs, powerset_tau=2, threshold=0.5)
        print('votes: ', votes)
        votes = votes.cpu().numpy()
        np.testing.assert_equal(votes.sum(), 5)

        votes = pick_votes_from_probabilities(
            probs=probs, powerset_tau=1, threshold=0.4)
        print('votes: ', votes)
        votes = votes.cpu().numpy()
        np.testing.assert_equal(votes.sum(), 4)


if __name__ == '__main__':
    unittest.main()
