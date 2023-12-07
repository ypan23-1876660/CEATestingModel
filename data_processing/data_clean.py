import numpy as np
import pandas as pd


from create_variable import create_var, combine_physid
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def data_impute(patient_dataframe):
    """
    Selecting columns in dataframe that are needed for the analysis
    Input: patient_dataframe (the patient_info and patient_visit merged dataframe) -of shape (76018,183)
    Output: Dataframe of shape (76018,33)
    """
    #note: since chances_of_recur is created based on value1 column, I'm not including that as a feature
    cols_to_select = ['PID', 'medctr2', 'age_deid', 'female', 'raceeth', 'partnered', 'lang', 'ins_medicare','ins_medicaid','ins_privatepay','ins_commercial','ins_other','charlson_wt',
 'memb_1yrprior','memb_at_dx','dx2membend','dx2membstart','dodi','dx2dod','IS_TOBACCO_USER','race_new','HISPANIC','charlson_wt_nocancer','BMI2',  'days_from_surveil',
 'days_from_last_visit','first_visit_from_surveil','cea_prev_visit','chances_of_recur','return_visit', 'value', 'physid_x','physid_y' ]
    patient_dataframe = patient_dataframe.loc[:, cols_to_select]
    return patient_dataframe

def remove_nan(patient_phys_info):
    """
    Dropping columns that have more than 30% data missing and dropping rows with missing values
    Input: patient_phy_info dataframe that included patient and physician characteristics of shape (569781, 54)
    Output: patient_phy_info dataframe that included patient and physician characteristics of shape (447057, 50)
    """
    nan_val_in_col = patient_phys_info.isna().sum()/len(patient_phys_info)
    #dropping all columns that have more than 30% of data is missing
    cols_to_drop = list(nan_val_in_col[nan_val_in_col>0.3].index)
    patient_phys_info.drop(cols_to_drop, axis = 1, inplace=True)
    #Since there are still some missing values in other columns but they represent <15% of data, I'm imputing those rows
    #which contain nan values as those values cannot be replaced by other approximations such as mean, median etc
    patient_phys_info.dropna(inplace=True)
    return patient_phys_info

def get_continous_columns(patient_phys_info):
    """
    Get the categorical columns in the patient_phys_info dataframe to be used in normalization of data 
    Input: patient_phy_info dataframe of shape (447057, 50)
    Output: Array of continuous_columns of length 26
    """
    continuous_columns = patient_phys_info.select_dtypes(include=['int64', 'float64']).columns
    # List of column names to drop
    columns_to_drop = ['raceeth', 'physician_gender','return_visit','memb_at_dx',
                        'memb_1yrprior','dodi', 'chances_of_recur']
    # Drop specified columns from continuous columns
    continuous_columns = continuous_columns.difference(columns_to_drop)
    return continuous_columns

def clean_rename_patPhyInfo(patient_phys_info):
    """
    Clean patient_phys_info by rename columns and dropping uninformative columns that will not be used in the model
    Input : patient_phys_info of shape (569781, 58)
    Output : patient_phys_info of shape (569781, 54)
    """
    patient_phys_info.rename(columns = {"female_x": "patient_gender", "female_y": "physician_gender"}, inplace = True)
    gender_map = {0: 'Male', 1: 'Female'}
    patient_phys_info['physician_gender'] = patient_phys_info['physician_gender'].map(gender_map)
    #I'm also dropping MEDCTR and JOB_TITLE, PID and physid
    patient_phys_info.drop(['MEDCTR','JOB_TITLE', 'PID', 'physid'], axis = 1, inplace = True)
    return patient_phys_info

def scale_patPhyInfo(patient_phys_info):
    """
    Scaling the categorical columns to be between 0 and 1 
    Input: patient_phys_info dataframe of shape (447057, 50)
    Output: patient_phys_info dataframe of shape (447057, 50)
    """
    #Extracting continous columns
    continuous_columns = get_continous_columns(patient_phys_info)
    # Extract continuous columns from the DataFrame
    continuous_data = patient_phys_info[continuous_columns]
    # Use MinMaxScaler to standardize between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(continuous_data)
    
    # Create a new DataFrame with the scaled values
    #Shape of scaled_data is (447057, 26)
    scaled_df = pd.DataFrame(scaled_data, columns=continuous_columns)
    #print(f'Shape of scaled_data is {scaled_df.shape}')
    #shape of categorical colimns is (447057, 24)
    #print(f'shape of categorical colimns is {patient_phys_info.drop(columns=continuous_columns).shape}')
    
    # Combine the non-continuous columns with the scaled continuous columns
    #Shape of patient_phys_info_scaled is (447057, 50)
    patient_phys_info_scaled = pd.concat([patient_phys_info.drop(columns=continuous_columns).reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
    
    return patient_phys_info_scaled

def encode_df(patient_phys_info):

    """
    Encoding categorical features
    Input: patient_phys_info dataframe of shape (447057, 50)
    Output: csv file that contains 447057 rows and 192 columns
    """

    #categorical variables (with multiple categories)
    #Not including medctr and  job title column
    categorical_columns = ["raceeth", "lang", "SPECIALTY", "MSOC_TYP_TX", "MSOC_BRD_NM", "PRVDR_SPCLTY", "ETHNICITY", 
                           "MSA", "race_new", "partnered", "patient_gender", "physician_gender","IS_TOBACCO_USER", "HISPANIC"]
    #getting unique values in these columns to see how many categories are there in each of these columns
    """print(f'Number of categorical colimn, {len(categorical_columns)}')
    for col in categorical_columns:
        print(f'Unique values in {col} is \n')
        print(len(patient_phys_info[col].unique()))"""
    
    ohencoder = OneHotEncoder(sparse=False)
    encoded_data = ohencoder.fit_transform(patient_phys_info[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=ohencoder.get_feature_names_out(categorical_columns))
   
    # Concatenate the encoded columns with the original DataFrame
    data_encoded = pd.concat([patient_phys_info.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Drop the original categorical columns if needed
    #Shape after encoding categorical colimns, (447057, 192)
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    #print(f'Shape after encoding categorical colimns, {data_encoded.shape}')
    
    # List of binary columns to encode
    binary_columns = ["ins_medicare", "ins_medicaid", "ins_privatepay",
                  "ins_commercial", "ins_other"]
    
    #Code block to check the number of different unique values in these columns
    """for col in binary_columns:
        print(f'Unique values in {col} is \n')
        print(patient_phys_info[col].unique())"""
    
    binary_map = {'Y': 1, "N": 0}
    for column in binary_columns:
        data_encoded[column] = data_encoded[column].map(binary_map)
    data_encoded.to_csv("../data/structured_info.csv")


if __name__ == "__main__":
    patient_visit = pd.read_csv("../data/deid_cea_v2.csv") # Patient revisit after surveillance 
    patient_info = pd.read_csv("../data/Final dataset prep_072521.csv") # All features data
    #patient shape is (76018,183)
    #Since there are few patients with 'value' == <1.0. Since our model cannot handle strings, I'm replacing <1.0 to 0.9 arbitrarily.
    val_to_replace = {'<1.0':'0.9', '< 1.0':'0.9', '>1500.00': '1500.1', '> 15000':'15001', '>15000.0':'15001'}
    patient_visit['value'] = patient_visit['value'].replace(val_to_replace)
    patient_visit['value'] = pd.to_numeric(patient_visit['value'])
    patient = create_var(patient_visit, patient_info)
    
    #patient_df_reduced shape is (76018,33)
    patient_df_reduced = data_impute(patient)
    #print(f"Shape of imputed patient_info data is {patient_df_reduced.shape}")

    # Read in physician data and combine all physician data
    directory_path = "../data/"
    physid_pattern = "*md*.csv"
    #Shape of phys_meta data is (76570, 27)
    phys_meta = combine_physid(directory_path, physid_pattern)
    #print(f"Shape of phys_meta data is {phys_meta.shape}")
    
    # Combine patient_meta and physician_meta to create a meta file for downstream
    #Shape of merged patient_phys_info is (569781, 58)
    patient_phys_info = patient_df_reduced.merge(phys_meta, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y']) # use "patient_phys_info" dataframe to continue working in dataframe
    #print(f"Shape of merged patient_phys_info is {patient_phys_info.shape}")
    
    #Shape of cleaned and renamed patient_phys_info is (569781, 54)
    patient_phys_info = clean_rename_patPhyInfo(patient_phys_info)
    #print(f"Shape of cleaned and renamed patient_phys_info is {patient_phys_info.shape}")

    #Shape of patient_phys_info after removing nan is (447057, 50)
    patient_phys_info = remove_nan(patient_phys_info) 
    #print(f"Shape of patient_phys_info after removing nan is {patient_phys_info.shape}")
    
    #Scaling continuous columns between 0 and 1
    #Shape of patient_phys_info_scaled is (447057, 50)
    patient_phys_info_scaled = scale_patPhyInfo(patient_phys_info)
    #print(f"Shape of patient_phys_info_scaled is {patient_phys_info_scaled.shape}")
    
    #Encoding categorical columns
    #Shape of encoded data, (447057, 192)
    encode_df(patient_phys_info_scaled)
    
    