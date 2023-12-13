""" This script includes test cases to test the correctness of functions provided in data_clean.py script """

import numpy as np
import pandas as pd

from ml4cea import create_var, combine_physid
from ml4cea import data_impute, remove_nan, clean_rename_patPhyInfo, scale_patPhyInfo, encode_df


import unittest 

patient_visit = "data/deid_cea_v2.csv"
patient_info = "data/Final dataset prep_072521.csv"
patient = create_var(patient_visit, patient_info)
patient = data_impute(patient)
directory = ("data")
pattern = ("*md*.csv")
phys = combine_physid(directory)
df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])

class TestDataImpute(unittest.TestCase):
    """Running unittest on data_impute"""
    def test_data_impute_smoke(self):
        """Smoke test: ensuring the data_impute function is working"""
        result = data_impute(patient)
        self.assertIsNotNone(result)

    def test_data_impute_edge1(self):
        """Edge test: Return ValueError when empty data frame is provided"""
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            result = data_impute(empty_data)

    def test_data_impute_edge2(self):
        """Edge test: Return ValueError when desired feature columns are 
        missing in the dataframe provided"""
        data_with_missingCol = patient.drop('raceeth', axis = 1)
        with self.assertRaises(ValueError):
            result = data_impute(data_with_missingCol)

    def test_data_impute_oneshot(self):
        """One shot test: Asserts that the resultant dataframe has columns equal to 33 """
        result = data_impute(patient)
        assert result.shape[1] == 33, "Columns in dataframe not equal to 33"

        
class TestRemoveNan(unittest.TestCase):
    """Running unittest on remove_nan"""

    def test_remove_nan(self):
        """Smoke test: ensuring the remove_nan function is working"""
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        result = remove_nan(clean_merged_df)
        self.assertIsNotNone(result)

    def test_remove_nan_edge1(self):
        """Edge test: Return ValueError when empty data frame is provided"""
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            result = remove_nan(empty_data)

    def test_remove_nan_edge2(self):
        """Edge test: Return Value Error when number of columns in input df npt equal to 54"""
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        missing_df = clean_merged_df.drop('age_deid', axis = 1)
        with self.assertRaises(ValueError):
            result = remove_nan(missing_df)

    def test_remove_nan_oneshot(self):
        """One shot : Asserts that the resultant dataframe has 50columns """
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        result = remove_nan(clean_merged_df)
        assert result.shape[1] == 50, "Columns in dataframe not equal to 50"
        
class TestCleanRenamepatPthyInfo(unittest.TestCase):
    
    def test_clean_rename_patPthyInfo(self):
        """Smoke test: ensuring the clean_rename_patPhyInfo function is working"""
        result = clean_rename_patPhyInfo(df)
        self.assertIsNotNone(result)

    def test_clean_rename_patPthyInfo_oneshot(self):
        """One shot test: Asserts that the resultant dataframe has columns equal to 54 """
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        result = clean_rename_patPhyInfo(merged_df)
        assert result.shape[1] == 54, "Columns in dataframe not equal to 54"

        
class TestScalepatPthyInfo(unittest.TestCase):

    def test_scale_pathPhyInfo(self):
        """Smoke test: ensuring the scale_pathPhyInfo function is working"""
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        full_df = remove_nan(clean_merged_df)
        result = scale_patPhyInfo(full_df)
        self.assertIsNotNone(result)
    
    def test_scale_pathPhyInfo_edge1(self):
        """Edge test: Raises ValueError if the number of continuous columns is not equal to 26"""
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        full_df = remove_nan(clean_merged_df).drop('days_from_last_visit',axis = 1)
        with self.assertRaises(ValueError):
            result = scale_patPhyInfo(full_df)
    
    def test_scale_pathPhyInfo_oneshot(self):
        """One shot test: Asserts that the number of rows of resultant dataframe is
         same as the input dataframe to scale_patPhyInfo() and number of columns in the
          resultant dataframe is 50 """
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        full_df = remove_nan(clean_merged_df)
        result = scale_patPhyInfo(full_df)
        assert result.shape[0] == full_df.shape[0] and result.shape[1] == 50 , "The shape of the resultant dataframe is incorrect"


class TestEncode_df(unittest.TestCase):
    """Smoke test: ensuring the encode_df function is working"""
    def test_encode_df(self):
        """Smoke test: ensuring the encode_df function is working"""
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        full_df = remove_nan(clean_merged_df)
        scale_df = scale_patPhyInfo(full_df)
        result = encode_df(scale_df)
        self.assertIsNotNone(result)
    
    def test_encode_df_onehot(self):
        """One hot test: Asserts that the number of rows of resultant dataframe is
         same as the input dataframe to encode()"""
        merged_df = patient.merge(phys, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y'])
        clean_merged_df = clean_rename_patPhyInfo(merged_df)
        full_df = remove_nan(clean_merged_df)
        scale_df = scale_patPhyInfo(full_df)
        result = encode_df(scale_df)
        assert result.shape[0] == scale_df.shape[0], "The shape of the resultant dataframe is incorrect"
