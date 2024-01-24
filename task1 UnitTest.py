# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 01:30:36 2024

@author: VMoiseienko
"""

import unittest
from task1 import count_islands 

class TestIslandCounter(unittest.TestCase):
    def test_case_1(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 1]
        ]
        result = count_islands(matrix)
        self.assertEqual(result, 2)

    def test_case_2(self):
        matrix = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ]
        result = count_islands(matrix)
        self.assertEqual(result, 3)

    def test_case_3(self):
        matrix = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ]
        result = count_islands(matrix)
        self.assertEqual(result, 2)

if __name__ == '__main__':
    unittest.main()