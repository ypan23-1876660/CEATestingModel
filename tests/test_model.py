#importing required packages for models.py test codes 

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from model import load_data, feature_select, model_train, scaling_test_features

import unittest 

class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        test_path = "../data/modeltestvalid.csv"
        result= load_data(test_path)
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_unnamed_column(self):
        path = "../data/modeltestvalid.csv"
        result = load_data(path)
        self.assertNotIn('Unnamed: 0', result.columns)
        
    def test_load_data_inccorrect(self):
        test_invalid_path = "../data/doesnotexist.csv"
        with self.assertRaises(ValueError) as context:
            load_data(test_invalid_path)
        
class TestFeatureSelect(unittest.TestCase):
    def test_feature_select(self):
        df = load_data("../data/modeltestvalid.csv")
        X_train, y_train = feature_select(df)
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_train, pd.DataFrame)
    
    def test_fail_feature_select(self):
        df=load_data("../data/modeltestfail.csv")
        with self.assertRaises(ValueError) as context:
            feature_select(df)

class TestModelTrain(unittest.TestCase):
    def test_model_train(self):
        df = load_data("../data/modeltestvalid.csv")
        X_train, y_train = feature_select(df)
        result = model_train(X_train, y_train)
        self.assertIsNotNone(result)
    
    def test_model_train_reproducibility(self):
        df = load_data("../data/modeltestvalid.csv")
        X_train, y_train = feature_select(df)
        models = [model_train(X_train, y_train) for _ in range(5)]
        first_model_coef = models[0].coef_
        for i in range(1,5):
            current_model_coef = models[1].coef_
            self.assertTrue((first_model_coef == current_model_coef).all())

class TestScalingTestFeatures(unittest.TestCase):
    def test_scaling_test_features(self):
        df = load_data("../data/modeltestvalid.csv")
        result = scaling_test_features(df)
        self.assertIsNotNone(result)
        
