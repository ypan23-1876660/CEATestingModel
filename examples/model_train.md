
Use the user input data to train the model, and export the trained model as pickel object 
-------------------------------------------------------------------------

Use the `scripts/model_train.py` to generate the trained model based on user input data. The trained model is saved as a pickle object and can be retrieved in the designated output path. 

1. Update the following arguments in `scripts/model_train.py`

  ```
  PATH = "data/"
  PATIENT_VISTI = "deid_cea_v2.csv" 
  PATIENT_INFOR = "Final dataset prep_072521.csv" 
  OUTPUT_PATH = "data/user_output/"
  MODEL_NAME = "model.pkl" 
  MODEL_PATH = os.path.join(OUTPUT_PATH, MODEL_NAME)
  ```
2. Run:`python scripts/model_train.py`
3. The function automatically exports the trained model named `MODEL_NAME` as a .pkl object to specified `OUTPUT_PATH`
----------
