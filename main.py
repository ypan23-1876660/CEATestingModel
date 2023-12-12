# import pandas as pd
# import numpy as np

# import glob
# from functools import reduce
# import sys
# import streamlit as st

#sys.path.append("ml4cea")
import os
import pickle
from ml4cea import create_var, combine_physid, data_impute, export_df, model_create, show_predict_page

# -------- USER INPUT ------- # 
PATH = "data/"
PATIENT_VISTI = "deid_cea_v2.csv"
PATIENT_INFOR = "Final dataset prep_072521.csv"
MODEL_NAME = "model.pkl"
MODEL_PATH = os.path.join(PATH, MODEL_NAME)
do_train = True # Change to False if want to use different data 
# --------------------------- #

if do_train:
    # Combine patient visit inforamtion and patient meta information 
    PATEINT_VISIT_PATH = os.path.join(PATH,PATIENT_VISTI)
    PATEINT_INFOR_PATH = os.path.join(PATH, PATIENT_INFOR)

    patient = create_var(PATEINT_VISIT_PATH, PATEINT_INFOR_PATH)

    # Select all the necessary features for model fitting 
    patient_df_reduced = data_impute(patient)

    # Get all the physician information and combine them into a meta dataframe
    phys_meta = combine_physid(PATH)

    # Combine patient and physician information 
    patient_phys_info = patient_df_reduced.merge(phys_meta, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y']) # use "patient_phys_info" dataframe to continue working in dataframe

    # Export proessed dataframe for modeling 
    cleaned_patient_phys_info = export_df(patient_phys_info)
    
    # Fit model and save 
    model_create(cleaned_patient_phys_info, MODEL_PATH)

with open(MODEL_PATH, 'rb') as file:
    # model = pickle.load(file)
    output = pickle.load(file)

model = output["model"]
min_train = output["min_train"]
max_train = output["max_train"]

# Create GUI
show_predict_page(model, min_train, max_train)
