"""
Use the trained model to predict based on the feature values 
"""
import os
import pickle
from ml4cea import create_var, combine_physid, data_impute, export_df, model_create, show_predict_page

# -------- USER INPUT ------- # 
OUTPUT_PATH = "data/output/" # Update this path if wanting to use other input data; default = default_output
MODEL_NAME = "model.pkl" # update this name if want to use different name 
MODEL_PATH = os.path.join(OUTPUT_PATH, MODEL_NAME)
# --------------------------- #

# Open model 
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Create GUI
show_predict_page(model, OUTPUT_PATH)
