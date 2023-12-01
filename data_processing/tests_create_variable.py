"""This is a test module which holds two smoke test"""

from functools import reduce
import glob
import pandas as pd
import numpy as np
import os
import unittest 
from create_variable import create_var
from create_variable import combine_physid

class TestCreateVar(unittest.TestCase):
    """Running unittest on create_var"""
    def test_create_var(self):
        """Smoke test: ensuring the dataframe is created"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv")
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv")
        result = create_var(patient_visit, patient_info)
        self.assertIsNotNone(result)


    """Running unittest on combine_physid"""
    def test_combine_physid(self):
        """Somke test: ensuring the dataframe is created"""
        directory = ("../data")
        pattern = ("*md*.csv")
        result = combine_physid(directory, pattern)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()