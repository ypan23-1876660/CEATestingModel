#importing required packages for models.py test codes 

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import os
import tempfile

from ml4cea import load_data, feature_select, model_train, scaling_test_features

import unittest 

class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        """Smoke test to ensure the function works."""
        test_path = "data/modeltestvalid.csv"
        result= load_data(test_path)
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_unnamed_column(self):
        """Oneshot test to ensure the no named column is deleted."""
        path = "data/modeltestvalid.csv"
        result = load_data(path)
        self.assertNotIn('Unnamed: 0', result.columns)
        
    def test_load_data_inccorrect(self):
        """Edge test to make sure the filepath works."""
        test_invalid_path = "data/doesnotexist.csv"
        with self.assertRaises(ValueError) as context:
            load_data(test_invalid_path)
        
class TestFeatureSelect(unittest.TestCase):
    def test_feature_select(self):
        """Smoke test to ensure the function works"""
        df = load_data("data/modeltestvalid.csv")
        X_train, y_train = feature_select(df)
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_train, pd.DataFrame)
    
    def test_fail_feature_select(self):
        """Edge test by adding a fail dataset to raise ValueError."""
        df=load_data("data/modeltestfail.csv")
        with self.assertRaises(ValueError) as context:
            feature_select(df)

class TestModelTrain(unittest.TestCase):
    def test_model_train(self):
        """Smoke test to make sure the function works."""
        df = load_data("data/modeltestvalid.csv")
        X_train, y_train = feature_select(df)
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        os.remove(self.temp_file.name)
        model_train(X_train, y_train, model_path=self.temp_file.name)
        
        file_size = os.path.getsize(self.temp_file.name)
        self.assertGreater(file_size, 0)
        
        with open(self.temp_file.name, 'rb') as file:
            loaded_output = pickle.load(file)
        self.assertIsInstance(loaded_output['model'], BaseEstimator)
            

class TestScalingTestFeatures(unittest.TestCase):
    def test_scaling_test_features(self):
        """Smoke test to make sure the function works."""
        df = load_data("data/modeltestvalid2.csv")
        min_train = np.load("data/default_output/min_train.npy")
        max_train = np.load("data/default_output/max_train.npy")
        result = scaling_test_features(df, min_train, max_train)
        self.assertIsNotNone(result)
      
        
