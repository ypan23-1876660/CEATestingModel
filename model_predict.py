import os
import pickle
import streamlit as st
import pandas as pd
from ml4cea import scaling_test_features

# -------- USER INPUT ------- # 
OUTPUT_PATH = "data/default_output/" 
MODEL_NAME = "model.pkl" 
MODEL_PATH = os.path.join(OUTPUT_PATH, MODEL_NAME)
# --------------------------- #

# Open model 
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# run app 
st.title("Predictive Healthcare: Machine Learning for Optimized CEA Testing in Colorectal Cancer Patients")
st.write("""### Predicting a patient's likelihood of being tested within the next three months""")
#st.write("""We need some patient information to predict the revisit""")

chances = (1,0)
days_from_last_visit = st.number_input("Enter the number of days from the last visit of the patient", value=None, placeholder="Type a number", min_value =0)
days_from_surveil = st.number_input("Enter the number of days from the patient's start of surveilliance",  value=None, placeholder="Type a number", min_value =0)
first_visit_from_surveil = st.number_input("Enter the number of days from the patient's first visit after surveilliance",  value=None, min_value =0, placeholder="Type a number")
cea_prev_visit = st.number_input("Enter the patient's Carcinoembryonic Antigen (CEA) value from the last visit",  value=None, placeholder="Type a number", format="%.0f")
chances_of_recur = st.selectbox("Recurrence of colorectal cancer, 1 is Recurrence, 0 is no Recurrence", chances)


predict_df = pd.DataFrame({"days_from_last_visit": [days_from_last_visit],
                "days_from_surveil": [days_from_surveil],
                "first_visit_from_surveil": [first_visit_from_surveil],
                "cea_prev_visit": [cea_prev_visit],
                "chances_of_recur": [chances_of_recur]
                })

ok = st.button("Predict")

if ok:
    predict_df_scaled = scaling_test_features(predict_df, OUTPUT_PATH)
    chance_of_return = model.predict(predict_df_scaled)
    if chance_of_return[0] == 1:
        st.subheader(f"The patient is likely to come back for the return visit")
    elif chance_of_return[0] == 0:
        st.subheader(f"The patient is not likely to come back for the return visit")
