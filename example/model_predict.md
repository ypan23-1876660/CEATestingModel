
Use the trained model to predict a patient's likelihood of being tested in the next three months
-------------------------------------------------------------------------

Remote to the website: [https://ceatestingmodel.streamlit.app/]

Update the following arguments in each of the text box on the website and click `Predict`


The page will returns with the statements based on the input values: 
- The patient is likely to come back for the return visit
- The patient is not likely to come back for the return visit 


-------------------------------------------------------------------------

To use the trained model based on the user input, update the following arguments in `model_predict.py`:

```
OUTPUT_PATH = "data/user_output/" 
MODEL_NAME = "model.pkl" # update this name if want to use different name 
MODEL_PATH = os.path.join(OUTPUT_PATH, MODEL_NAME)
```

The function will use the specified model `MODEL_NAME` to predict a patient's likelihood of being tested in the next three months.

Then run:
`streamlit run model_predict.py`



