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
        
    def test_days_from_surveil(self):
        """Oneshot test: variable output is correct"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv") 
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv")
        df = create_var(patient_visit, patient_info)
        result = df['days_from_surveil']
        value = df['dx2cea'] - df['dx2surveildate']
        self.assertEqual(result.all(), value.all())
        
    def test_days_from_visit(self):
        """Oneshot test: variable output is correct"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv") 
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv") 
        df = create_var(patient_visit, patient_info)
        result = df['days_from_surveil']
        value = df['dx2cea'].diff()
        self.assertEqual(result.all(), value.all())
  
    def test_first_visit(self):
        """Oneshot test: variable output is correct"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv") 
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv") 
        df = create_var(patient_visit, patient_info)
        result = df['first_visit_from_surveil']
        value = df.groupby('PID').head(1)['dx2cea'] - df.groupby('PID').head(1)['dx2surveildate']
        value2 = value.ffill()
        self.assertEqual(result.all(), value2.all())
        
    def test_previous_visit(self):
        """Oneshot test: variable output is correct"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv") 
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv") 
        df = create_var(patient_visit, patient_info)
        result = df['cea_prev_visit']
        expected = df['value'].shift(1)
        self.assertEqual(result.all(), expected.all())
        
    def test_reoccurance(self):
        """Oneshot test: variable output is correct"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv") 
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv") 
        df = create_var(patient_visit, patient_info)
        result = df['chances_of_recur']
        expected1 = df['value'].astype(str)
        expected2 = expected1.str.replace('<', '').str.replace('>', '')
        expected3 = pd.to_numeric(expected2, errors = 'coerce')
        expected4 = np.where(expected3 > 5, 1, 0)
        self.assertEqual(result.all(), expected4.all())
       
    def test_return(self):
        """Oneshot test: variable output is correct"""
        patient_visit = pd.read_csv("../data/deid_cea_v2.csv") 
        patient_info = pd.read_csv("../data/Final dataset prep_072521.csv") 
        df = create_var(patient_visit, patient_info)
        result = df['return_visit'] 
        value = np.where(df.groupby('PID')['dx2cea'].diff() <= 90, 1, 0)
        self.assertEqual(result.all(), value.all())

        
class TestCombinePhysid(unittest.TestCase):
    """Running unittest on combine_physid"""
    def test_combine_physid(self):
        """Somke test: ensuring the dataframe is created"""
        directory = ("../data")
        pattern = ("*md*.csv")
        result = combine_physid(directory, pattern)
        self.assertIsNotNone(result)
        
