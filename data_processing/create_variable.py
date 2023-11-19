import pandas as pd
import numpy as np
import os

### OUTPU: 'META' DATAFRAME FOR THE FOLLOWING STPES

## load in data 
patient_visit = pd.read_csv("../data/deid_cea_v2.csv") # Patient revisit after surveillance 
patient_meta = pd.read_csv("../data/Final dataset prep_072521.csv") # All features data

## Merging patient visit and patient meta
meta = patient_meta.merge(patient_visit,on="PID", how='outer').sort_index()

## Merging patient visit and patient meta
meta = patient_meta.merge(patient_visit,on="PID", how='outer').sort_index()

##creating new value variable (value1) where the values is float
meta['value1'] = meta['value'].astype(str)
meta['value1'] = meta['value1'].str.replace('<', '').str.replace('>', '')
meta['value1'] = pd.to_numeric(meta['value1'], errors = 'coerce')


## Creating all variables 
# How long from the start of surveillance
meta['days_from_surveil'] = meta['dx2cea'] - meta['dx2surveildate'] 

# How long from the previous visit
meta['days_from_last_visit'] = meta['dx2cea'].diff() 

# How long from the start of surveillance to their first visit
meta['first_visit_from_surveil'] = meta.groupby('PID').head(1)['dx2cea'] - meta.groupby('PID').head(1)['dx2surveildate']
meta['first_visit_from_surveil'] = meta['first_visit_from_surveil'].ffill()

# CEA value from previous visit 
meta['cea_prev_visit'] = meta['value1'].shift(1)

# If CEA value >5, then higher chance of reoccurrence
meta['chances_of_recur'] = np.where(meta['value1'] > 5, 1, 0)

# Patient revisits every 90 days 
meta['return_visit'] = np.where(meta.groupby('PID')['dx2cea'].diff() <= 90, 1, 0)