import numpy as np
import pandas as pd

from create_variable import create_var, combine_physid
from data_clean import data_impute, remove_nan, clean_rename_patPhyInfo, scale_patPhyInfo, encode_df
from sklearn.preprocessing import OneHotEncoder

import unittest 

patient_visit = pd.read_csv("../data/deid_cea_v2.csv")
patient_info = pd.read_csv("../data/Final dataset prep_072521.csv")
patient = create_var(patient_visit, patient_info)
directory = ("../data")
pattern = ("*md*.csv")
phys = combine_physid(directory, pattern)
df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])


class TestDataImpute(unittest.TestCase):
    """Running unittest on data_impute"""
    def test_data_impute(self):
        """Smoke test: ensuring the data_impute function is working"""
        result = data_impute(patient)
        self.assertIsNotNone(result)
    
        
class TestRemoveNan(unittest.TestCase):
    """Running unittest on remove_nan"""
    def test_remove_nan(self):
        """Smoke test: ensuring the remove_nan function is working"""
        result = remove_nan(df)
        self.assertIsNotNone(result)
        
class TestCleanRenamepatPthyInfo(unittest.TestCase):
    """Smoke test: ensuring the clean_rename_patPhyInfo function is working"""
    def test_clean_rename_patPthyInfo(self):
        result = clean_rename_patPhyInfo(df)
        self.assertIsNotNone(result)
        
class TestScalepatPthyInfo(unittest.TestCase):
    """Smoke test: ensuring the scale_pathPhyInfo function is working"""
    def test_scale_pathPhyInfo(self):
        result = scale_patPhyInfo(df)
        self.assertIsNotNone(result)

class TestEncode_df(unittest.TestCase):
    """Smoke test: ensuring the encode_df function is working"""
    def test_encode_df(self):
        result = encode_df(df)
        self.assertIsNotNone(result)
        
        