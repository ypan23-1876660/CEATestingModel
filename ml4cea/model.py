# importing required packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os


def load_data(data_path):
    """
    Load patient and physician information data from a CSV file.

    Args:
        - data_path (str): Path to the CSV file containing patient and physician information.

    Returns:
        - patient_phy_data (pd.DataFrame): Loaded data as a Pandas DataFrame.
    Raises:
        - ValueError: If the specified file does not exist.
    """
    if os.path.exists(data_path):
        patient_phy_data = pd.read_csv(data_path)
        patient_phy_data.drop(['Unnamed: 0'], axis=1, inplace=True)
        return patient_phy_data
    else:
        raise ValueError(f"This file {data_path} does not exist.")


def feature_select(patient_phy_data):
    """
    Select relevant features for model training.

    Args:
        - patient_phy_data (pd.DataFrame): DataFrame containing patient and physician information.

    Returns:
        - X_train (pd.DataFrame): Features for training the model.
        - y_train (pd.Series): Response variable for training the model.
    Raises:
        - ValueError: If predictors or response variables are missing in the dataset.
    """
    predictors = ['days_from_last_visit', 'days_from_surveil', 'first_visit_from_surveil', 'cea_prev_visit', 'chances_of_recur']
    response = ['return_visit']

    # Checking if all the predictors are present in patient_phy_data
    missing_predictors = [col for col in predictors if col not in patient_phy_data.columns]
    if missing_predictors:
        raise ValueError(f"The following predictors are missing in the dataset: {', '.join(missing_predictors)}")
    else:
        pass

    # Checking if the response is present in patient_phy_data
    missing_response = [col for col in response if col not in patient_phy_data.columns]
    if missing_response:
        raise ValueError(f"The response is missing in the dataset: {', '.join(missing_response)}")
    else:
        pass

    X = patient_phy_data[predictors]
    y = patient_phy_data[response]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train



def model_train(X_train,y_train, model_path = "data/default_output/model.pkl"):

    """
    Train a logistic regression model and save it to a specified file.

    Args:
        - X_train (pd.DataFrame): Features for training the model.
        - y_train (pd.Series): Response variable for training the model.
        - model_path (str): Path to save the trained model.

    Returns:
        - None
    """

    model_cv = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10], cv=5, solver='liblinear')
    model_cv.fit(X_train, y_train.values.ravel())
    with open(model_path, 'wb') as file:
        pickle.dump(model_cv, file)


def scaling_test_features(predict_df, output_path='data/default_output/'):
    """
    Scale test features by MinMaxScaler
    :param predict_df: pd.DataFrame
        test dataframe with user info as columns
    :param output_path: path
        path where min_train.npy and max_train.npy is stored. Default is the default_output
    returns: scaled_df: pd.DataFrame
        test dataframe scaled by MinMaxScaler 
    """
    data = predict_df.to_numpy()
    columns = predict_df.columns.tolist()
    max_train_path = os.path.join(output_path, "max_train.npy")
    min_train_path = os.path.join(output_path, "min_train.npy")
    max_train = np.load(max_train_path)
    min_train = np.load(min_train_path)

    def MinMaxScalerByValue(X, max_train, min_train):
        X_std = (X - min_train) / (max_train - min_train)
        X_scaled = X_std * (max_train - min_train) + min_train
        return X_scaled

    scaled_data = MinMaxScalerByValue(data, max_train, min_train)
    scaled_df = pd.DataFrame({columns[i]: scaled_data[:, i] for i in range(len(columns))})
    return scaled_df


def model_create(structured_info, model_path = "data/default_output/model.pkl"):
    X_train, y_train = feature_select(structured_info)
    model_train(X_train, y_train, model_path)

