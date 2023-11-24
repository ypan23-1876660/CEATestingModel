import pandas as pd
import numpy as np
import os
import glob
from functools import reduce

### OUTPUT: 'META' DATAFRAME FOR THE FOLLOWING STPES

## Merging patient visit and patient info
def create_var(patient_visit, patient_info):
    '''A function that takes in deid_cea_v2.csv and Final dataset prep_072521.csv to
    create all necessary variables. 
    Input those the two files and will return a combined file with all the new variables.
    '''
    ## Merging patient visit and patient meta
    df = patient_info.merge(patient_visit,on="PID", how='outer').sort_index()

    ## Creating all variables 
    # How long from the start of surveillance
    df['days_from_surveil'] = df['dx2cea'] - df['dx2surveildate'] 

    # How long from the previous visit
    df['days_from_last_visit'] = df['dx2cea'].diff() 

    # How long from the start of surveillance to their first visit
    df['first_visit_from_surveil'] = df.groupby('PID').head(1)['dx2cea'] - df.groupby('PID').head(1)['dx2surveildate']
    df['first_visit_from_surveil'] = df['first_visit_from_surveil'].ffill()

    # CEA value from previous visit 
    df['cea_prev_visit'] = df['value'].shift(1)
    
    # If CEA value >5, then higher chance of reoccurrence
    # First creating new value variable (value1) to convert all values into numeric to perform logic 
    df['value1'] = df['value'].astype(str)
    df['value1'] = df['value1'].str.replace('<', '').str.replace('>', '')
    df['value1'] = pd.to_numeric(df['value1'], errors = 'coerce')
    df['chances_of_recur'] = np.where(df['value1'] > 5, 1, 0)

    # Patient revisits every 90 days 
    df['return_visit'] = np.where(df.groupby('PID')['dx2cea'].diff() <= 90, 1, 0)
    return df


def combine_physid(directory, pattern):
    """A function that reads in all the physician characteristics files *md*.csv. 
    Then comibne all the physical characteristics files. 
    """
    # Use glob to find all files matching the pattern
    file_pattern = os.path.join(directory, pattern)
    csv_files = glob.glob(file_pattern)

    # Create an empty list to store individual DataFrames
    dataframes = []

    # Read each CSV file and append its contents to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='windows-1254')
        dataframes.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_data = reduce(lambda x, y: pd.merge(x, y, on = 'physid'), dataframes)
    return combined_data

