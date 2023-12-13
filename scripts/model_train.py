"""
Use the input data to train the model 
"""
import os
import pickle
from ml4cea import create_var, combine_physid, data_impute, export_df, model_create, show_predict_page

# -------- USER INPUT ------- # 
PATH = "data/"
PATIENT_VISTI = "deid_cea_v2.csv" 
PATIENT_INFOR = "Final dataset prep_072521.csv" 
OUTPUT_PATH = "data/default_output/" # Update this path if wanting to use other input data; default = default_output
MODEL_NAME = "model.pkl" # update this name if want to use different name 
MODEL_PATH = os.path.join(OUTPUT_PATH, MODEL_NAME)
# --------------------------- #
# create a new folder if using user data 
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

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
cleaned_patient_phys_info = export_df(patient_phys_info, OUTPUT_PATH)

# Fit model and save the model in data folder
model_create(cleaned_patient_phys_info, MODEL_PATH)
