#importing required packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle


def feature_select(patient_phy_data):
    predictors = ['days_from_last_visit', 'days_from_surveil', 'first_visit_from_surveil', 'cea_prev_visit', 'chances_of_recur']
    response = ['return_visit']

    # checking if all the predictors are present in patient_phy_data
    missing_predictors = [col for col in predictors if col not in patient_phy_data.columns]
    if missing_predictors:
        raise ValueError(f"The following predictors are missing in the dataset: {', '.join(missing_predictors)}")
    else:
        pass 
    
    # checking if the response is present in patient_phy_data
    missing_response = [col for col in response if col not in patient_phy_data.columns]
    if missing_response:
        raise ValueError(f"The respomse is missing in the dataset: {', '.join(missing_response)}")
    else:
        pass

    X = patient_phy_data[predictors] 
    y = patient_phy_data[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train


def model_train(X_train,y_train, model_path = 'data/model.pkl'):
    model_cv = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10], cv=5, solver='liblinear')
    model_cv.fit(X_train,y_train.values.ravel())
    min_train = np.load("data/min_train.npy")
    max_train = np.load("data/max_train.npy")
    output = {"model": model_cv,
              "min_train": min_train,
              "max_train": max_train
              }
    with open (model_path, 'wb') as file:
        pickle.dump(output, file)
        # pickle.dump(model_cv, file)

def scaling_test_features(predict_df, max_train, min_train):
    """
    Scale test features by MinMaxScaler.

    Args:
        - predict_df (pd.DataFrame): test dataframe with user info as columns
    Returns:
        - scaled_df (pd.DataFrame): test dataframe scaled by MinMaxScaler.
    """
    data = predict_df.to_numpy()
    columns = predict_df.columns.tolist()

    def MinMaxScalerByValue(X, max_train, min_train):
        X_std = (X - min_train) / (max_train - min_train)
        X_scaled = X_std * (max_train - min_train) + min_train
        return X_scaled
    scaled_data = MinMaxScalerByValue(data, max_train, min_train)

    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame({columns[i]: scaled_data[:, i] for i in range(len(columns))})
    return scaled_df


def model_create(structured_info, model_path):
    #import pdb; pdb.set_trace()
    X_train, y_train = feature_select(structured_info)
    model_train(X_train, y_train, model_path)
