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
MODEL_PATH = "./data/model.pkl"
# --------------------------- #

with open(MODEL_PATH, 'rb') as file:
    # model = pickle.load(file)
    output = pickle.load(file)

model = output["model"]
min_train = output["min_train"]
max_train = output["max_train"]

# Create GUI
show_predict_page(model, min_train, max_train)
