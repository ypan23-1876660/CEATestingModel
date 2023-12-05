import numpy as np
import pandas as pd


from create_variable import create_var, combine_physid
from sklearn.preprocessing import OneHotEncoder


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
    Input: patient_phy_info dataframe that included patient and physician characteristics of shape (569781, 58)
    Output: patient_phy_info dataframe that included patient and physician characteristics of shape (445136, 54)
    """
    nan_val_in_col = patient_phys_info.isna().sum()/len(patient_phys_info)
    #dropping all columns that have more than 30% of data is missing
    cols_to_drop = list(nan_val_in_col[nan_val_in_col>0.3].index)
    patient_phys_info.drop(cols_to_drop, axis = 1, inplace=True)
    #Since there are still some missing values in other columns but they represent <15% of data, I'm imputing those rows
    #which contain nan values as those values cannot be replaced by other approximations such as mean, median etc
    patient_phys_info.dropna(inplace=True)
    return patient_phys_info

def encode_df(patient_phys_info):

    """
    Encoding categorical features
    Input: patient_phys_info dataframe of shape (445136, 54)
    Output: csv file that contains 445136 rows and  189 columns
    """

    #categorical variables (with multiple categories)
    #Not including medctr and  job title column
    categorical_columns = ["raceeth", "lang", "SPECIALTY", "MSOC_TYP_TX", "MSOC_BRD_NM", "PRVDR_SPCLTY", "ETHNICITY", 
                           "MSA", "race_new", "partnered", "patient_gender", "physician_gender","IS_TOBACCO_USER", "HISPANIC"]
    #getting unique values in these columns to see how many categories are there in each of these columns
    """for col in categorical_columns:
        print(f'Unique values in {col} is \n')
        print(patient_phys_info[col].unique())"""
    ohencoder = OneHotEncoder(sparse=False)
    encoded_data = ohencoder.fit_transform(patient_phys_info[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=ohencoder.get_feature_names_out(categorical_columns))
   
    # Concatenate the encoded columns with the original DataFrame
    data_encoded = pd.concat([patient_phys_info.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Drop the original categorical columns if needed
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    #I'm also dropping MEDCTR and JOB_TITLE
    data_encoded.drop(['MEDCTR','JOB_TITLE'], axis = 1, inplace = True)

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
    
    print(data_encoded.shape)
    
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

    # Read in physician data and combine all physician data
    directory_path = "../data/"
    physid_pattern = "*md*.csv"
    phys_meta = combine_physid(directory_path, physid_pattern)
    # Combine patient_meta and physician_meta to create a meta file for downstream
    patient_phys_info = patient_df_reduced.merge(phys_meta, how="left", left_on="physid_x", right_on="physid").drop(columns=['physid_x', 'physid_y']) # use "patient_phys_info" dataframe to continue working in dataframe
    patient_phys_info.rename(columns = {"female_x": "patient_gender", "female_y": "physician_gender"}, inplace = True)
    gender_map = {0: 'Male', 1: 'Female'}
    patient_phys_info['physician_gender'] = patient_phys_info['physician_gender'].map(gender_map)
    #patient_phys_info.to_csv("../data/patient_phys_info.csv")
    #shape of patient_phys_info before remove_nan is (569781, 58)
    #dropping four columns here namely ['dx2dod', 'MSOC_SCHL_SPCLT_TX', 'MSOC_DGR_ERN_TX', 'termination_yr']
    #and dropping rows that contain nan values
    #shape of patient_phys_info after remove_nan is (445136, 54)
    patient_phys_info = remove_nan(patient_phys_info) 
    encode_df(patient_phys_info)
    