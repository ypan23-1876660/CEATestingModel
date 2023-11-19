import pandas as pd
import numpy as np
import os
import glob
from functools import reduce
import sys

sys.path.append("../data_processing/")
from data_processing import create_var, combine_physid

### Data Processing
# Read in data
patient_visit = pd.read_csv("../data/deid_cea_v2.csv") # Patient revisit after surveillance 
patient_meta = pd.read_csv("../data/Final dataset prep_072521.csv") # All features data

# Read in physician data and combine all physician data
directory_path = "../data/"
physid_pattern = "*md*.csv"
phys_meta = combine_physid(directory_path, physid_pattern)

# Combine patient_meta and physician_meta to create a meta file for downstream
meta = create_var(patient_visit, patient_meta)
meta.merge(phys_meta, how="left", left_on=["physid_y"], right_on=["physid"])
