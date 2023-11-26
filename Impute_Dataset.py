import pandas as pd
import numpy as np
import os
import pandas as pd

from fancyimpute import IterativeImputer
from functools import reduce

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

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

patient_phys_info.head()
### Clean up Nans
### Model
### Model fitting

#Exploratory Data Analysis
#Exploratory data analysis should describe the missingness
patient_phys_info.isnull().values.any()

#Number of missing values for each variable
mis_val_var_specific=(patient_phys_info.isnull().sum())

pd.set_option('display.max_rows', None)  # Set to None to display all rows

#Unique number of patients
unique_patient_ids = patient_phys_info['PID'].nunique()

# Calculate the sum of the number of rows
num_rows_df = patient_phys_info.shape[0]

#Complete-Case Analysis
complete_cases_count = patient_phys_info.dropna().count()

# Columns to be removed
#'rt_sx', 'earlychemoname', 'earlychemorecur', and 'as.numeric.NA
columns_to_remove = ['rt_sx', 'earlychemoname', 'earlychemorecur','as.numeric.NA.']

# Drop specified columns
df_removed = patient_phys_info.drop(columns=columns_to_remove)


from sklearn.impute import SimpleImputer

# simple imputation using IterativeImputer
imp = SimpleImputer(strategy="most_frequent")
#print('Number of missing values for each variable:')
pd.set_option('display.max_rows', None)  
#print(imp.fit_transform(df_removed))

# Check for missing values in each column
missing_values = df_removed.isnull().sum()

# Print the variables with missing values
print("Variables with missing values:")
print(missing_values[missing_values > 0])

 # Using iloc[0] to get the first mode if multiple modes exist
df_filled_mode = df_removed.fillna(df_removed.mode().iloc[0]) 

#Exploratory Data Analysis _ Imputated Dataset
df_filled_mode.isnull().values.any()


