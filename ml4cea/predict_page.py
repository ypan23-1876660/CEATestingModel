import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# sys.path.append("../Models/")
# sys.path.append("../data/")
from ml4cea.model import scaling_test_features


def show_predict_page(model, output_path):
    """
    Show the website for predicting if the patient is likely to come back for return visit within the next three months. 
    Return:
        0: The patient is not likely to come back for the return visit 
        1: The patient is likely to come back for the return visit
    Args:
        - model: Trained model for predicting patient revisit.
        - output_path (str): Path to the output folder.
    """
    st.title("Prediction of patient's revisit within the next three months after the diagnosis of colorectal cancer")
    st.write("""### We need some patient information to predict the revisit""")

    chances = (1, 0)
    days_from_last_visit = st.number_input("Enter the number of days from the last visit of the patient", value=None, placeholder="Type a number", min_value=0)
    days_from_surveil = st.number_input("Enter the number of days from the patient's start of surveillance", value=None, placeholder="Type a number", min_value=0)
    first_visit_from_surveil = st.number_input("Enter the number of days from the patient's first visit after surveillance", value=None, min_value=0, placeholder="Type a number")
    cea_prev_visit = st.number_input("Enter the number of days of the patient's CEA value from the last visit", value=None, placeholder="Type a number", format="%.0f")
    chances_of_recur = st.selectbox("Reoccurrence of colorectal cancer, 1 is reoccurrence, 0 is no reoccurrence", chances)

    predict_df = pd.DataFrame({
        "days_from_last_visit": [days_from_last_visit],
        "days_from_surveil": [days_from_surveil],
        "first_visit_from_surveil": [first_visit_from_surveil],
        "cea_prev_visit": [cea_prev_visit],
        "chances_of_recur": [chances_of_recur]
    })

    ok = st.button("Predict")

    if ok:
        predict_df_scaled = scaling_test_features(predict_df, output_path)
        chance_of_return = model.predict(predict_df_scaled)
        if chance_of_return[0] == 1:
            st.subheader(f"The patient is likely to come back for the return visit")
        elif chance_of_return[0] == 0:
            st.subheader(f"The patient is not likely to come back for the return visit")

