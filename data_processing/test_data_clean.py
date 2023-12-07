import numpy as np
import pandas as pd

from create_variable import create_var, combine_physid
from data_clean import data_impute, remove_nan, encode_df
from sklearn.preprocessing import OneHotEncoder

import unittest 

class TestDataImpute(unittest.TestCase):
    """Running unittest on data_impute"""
    def test_data_impute(self):
        """Smoke test: ensuring the data_impute function is working"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv")
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv")
        df = create_var(patient_visit, patient_info)
        result = data_impute(df)
        self.assertIsNotNone(result)
    
        
class TestRemoveNan(unittest.TestCase):
    """Running unittest on remove_nan"""
    def test_remove_nan(self):
        """Smoke test: ensuring the data inpute function is working"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv")
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv")
        patient = create_var(patient_visit, patient_info)
        directory = ("../data")
        pattern = ("*md*.csv")
        phys = combine_physid(directory, pattern)
        df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        result = remove_nan(df)
        self.assertIsNotNone(result)
        