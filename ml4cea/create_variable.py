"""
Step1: Merge all the necessary files into meta dataframe and create necessary variables for modeling
"""
import pandas as pd
import numpy as np
import os
import glob
from functools import reduce

def create_var(patient_visit, patient_info):
    '''
    A function that merge deid_cea_v2.csv and Final dataset prep_072521.csv to
    create all necessary variables/features for ML model. 

    : param patient_visit: csv
        deid_cea_v2.csv in /data. This file contains the patient information including
        patient ID, CEA value, days from last visit, and physician ID. 
    : param patient_info: csv
        Final dataset prep_072521.csv in /data. This file is the meta data with all 
        the patient information 
    : return: merged pandas.DataFrame
        Combined patient information with newly created variables: days_from_surveil, first_visit_from_surveil,
        cea_prev_visit, chances_of_recur, return_visit
    '''
    # Read in patient visit information and patient meta information 
    patient_visit = pd.read_csv(patient_visit)
    patient_info = pd.read_csv(patient_info)
    
    # Merging patient visit and patient meta
    df = patient_info.merge(patient_visit,on="PID", how='outer').sort_index()

    ## Creating all variables 
    # How long from the start of surveillance
    df['days_from_surveil'] = df['dx2cea'] - df['dx2surveildate'] 

    # How long from the previous visit
    df['days_from_last_visit'] = df['dx2cea'].diff() 

    # How long from the start of surveillance to their first visit
    df['first_visit_from_surveil'] = df.groupby('PID').head(1)['dx2cea'] - df.groupby('PID').head(1)['dx2surveildate']
    df['first_visit_from_surveil'] = df['first_visit_from_surveil'].ffill()

    # If CEA value >5, then higher chance of reoccurrence
    # First creating new value variable (value1) to convert all values into numeric to perform logic 
    df['value1'] = df['value'].astype(str)
    df['value1'] = df['value1'].str.replace('<', '').str.replace('>', '')
    df['value1'] = pd.to_numeric(df['value1'], errors = 'coerce')
    #Since there are few patients with 'value' == <1.0. Since our model cannot handle strings, Replacing <1.0 to 0.9 arbitrarily.
    val_to_replace = {'<1.0':'0.9', '< 1.0':'0.9', '>1500.00': '1500.1', '> 15000':'15001', '>15000.0':'15001'}
    df['value1'] = df['value1'].replace(val_to_replace)
    df['value'] = pd.to_numeric(df['value1'])
    
    # CEA value from previous visit 
    df['cea_prev_visit'] = df['value'].shift(1)

    # chances of reccurance 
    df['chances_of_recur'] = np.where(df['value'] > 5, 1, 0)
    # Patient revisits every 90 days 
    df['return_visit'] = np.where(df.groupby('PID')['dx2cea'].diff() <= 90, 1, 0)
    return df


def combine_physid(directory):
    '''
    A function that merge all the files with physician information in /data 

    : param directory: path
        /data path in the repository
    : return: merged pandas.DataFrame
        Combine all physicain information in to a meta dataframe
    '''
    pattern = "*md*.csv" # Get all the physician ids with pattern
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
