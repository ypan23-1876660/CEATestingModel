import pandas as pd
import numpy as np
import os

# load in data 
data = pd.read_excel("../data/deid_cea_v2.xlsx")

# Create a new dataframe for condensed patient information 
new_df = pd.DataFrame()

# Calculate the probability of the patient consistenly returning every three months (90 days)
data['return'] = np.where(data.groupby('PID')['dx2cea'].diff() <= 90, 1, 0)
new_df['return_probability'] = data.groupby('PID')['return'].mean()
new_df = new_df.reset_index()

new_df.to_csv('../data/deid_cea_v2_condensed.csv', index=False)