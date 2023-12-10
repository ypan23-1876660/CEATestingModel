#importing required packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os


def load_data(data_path):
    if os.path.exists(data_path):
        patient_phy_data = pd.read_csv(data_path)
        patient_phy_data.drop(['Unnamed: 0'], axis = 1, inplace=True)
        return patient_phy_data
    else:
        raise ValueError(f"This file {data_path} does not exist.")



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
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train


def model_train(X_train,y_train):
    model_cv = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10], cv=5, solver='liblinear')
    model_cv.fit(X_train,y_train)
    with open ('../gui/model.pkl', 'wb') as file:
        pickle.dump(model_cv, file)

def scaling_test_features(predict_df):
    """
    Scale test_features
    """
    continous_predictors = ['days_from_last_visit', 'days_from_surveil', 'first_visit_from_surveil', 'cea_prev_visit']
     # Extract continuous columns from the DataFrame
    continuous_data = predict_df[continous_predictors]
    # Use MinMaxScaler to standardize between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(continuous_data)
    scaled_df = pd.DataFrame(scaled_data, columns=continous_predictors)
    predict_df_scaled = pd.concat([predict_df.drop(columns=continous_predictors).reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
    return predict_df_scaled
    #y_prediction = model_cv.predict(predict_df_scaled)

if __name__ == "__main__":
    data_path = "../data/structured_info.csv"
    patient_phy_data = load_data(data_path)
    print(patient_phy_data.shape)
    X_train, y_train = feature_select(patient_phy_data)
    model_train(X_train, y_train)


