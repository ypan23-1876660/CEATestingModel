
Use the trained model to predict a patient's likelihood of being tested in the next three months
-------------------------------------------------------------------------

1. Remote to the website: [https://ceatestingmodel.streamlit.app/]
2. Update the following arguments in each of the text box on the website and click `Predict`
3. The page will returns with the statements based on the input values: 
- The patient is likely to come back for the return visit
- The patient is not likely to come back for the return visit <br>
<br>
Website demo:
<br>


![example_model_predict](https://github.com/ML4CEA/CEATestingModel/assets/62965045/9c93c03e-71b5-4129-8696-6c4381ebee20)


Use the trained model based on the user input
-------------------------------------------------------------------------

1. Update the following arguments in `model_predict.py`:
  ```
  OUTPUT_PATH = "data/user_output/" 
  MODEL_NAME = "model.pkl" 
  MODEL_PATH = os.path.join(OUTPUT_PATH, MODEL_NAME)
  ```
The function will use the specified model `MODEL_NAME` to predict a patient's likelihood of being tested in the next three months

2. Run: `streamlit run model_predict.py`
3. This will direct user to the website where user can type in the input values to get the prediction of a patient's likelihood of being tested in the next three months



