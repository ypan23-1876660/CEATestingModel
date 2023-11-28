import pandas as pd
import numpy as np
import os
import glob
from functools import reduce
import sys

sys.path.append("./data_processing/")
from create_variable import create_var, combine_physid

### Data Processing
# Read in data
patient_visit = pd.read_csv("./data/deid_cea_v2.csv") # Patient revisit after surveillance 
patient_info = pd.read_csv("./data/Final dataset prep_072521.csv") # All features data
patient = create_var(patient_visit, patient_info)

# Read in physician data and combine all physician data
directory_path = "./data/"
physid_pattern = "*md*.csv"
phys_meta = combine_physid(directory_path, physid_pattern)

# Combine patient_meta and physician_meta to create a meta file for downstream
patient_phys_info = patient.merge(phys_meta, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y']) # use "patient_phys_info" dataframe to continue working in dataframe
patient_phys_info.to_csv("./data/patient_phys_info.csv") # use "patient_phys_info.csv" to read in the information and continue working in another file

### Clean up Nans
### Model
### Model fitting
