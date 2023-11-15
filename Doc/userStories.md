## User stories:

**Goal**

The predictions from this model itself will be inputs into a larger simulation model where we are trying to simulate outcomes under a guideline-based surveillance schedule, where guidelines say patients should come in every 3-6 months, and that assessment of whether the patient comes in for testing is driven by several factors, which is what we are trying to figure out. 


## Users:
**Researchers (others)**:
Researchers are going to use the pre-trained model to predict colorectal cancer patients' clinic visit/CEA testing outcomes within their own datasets. They will be able to input colorectal cancer patient data for specific variables in an interactive interface to get the output which is the outcome of the prediction. The researchers will either be skilled coder who can change the machine learning model (bases of the software), if they choose to change it to cater their needs. Or they could not have a lot of coding skills and could only be interacting with the interface of the software so they are only getting the prediction of clinic visit/CEA testing outcomes. 

**Policy maker (insurance companies, health plan makers)**:
They are going to be using the interface to pertain the prediction of colorectal cancer patients' clinical visits/CEA testing outcomes for given groups in order to implement an intervention.
They will only be interacting with the interface- they will  be inputting the necessary information needed to provide the prediction. They will have limited technical (computing and modeling) expertise, therefore, the interface needs to be user-friendly. 

**Physicians/Doctors**: 
Physicians will be using this to cater to their colorectal cancer patients needs- to understand the likelihood of patients coming in for visit/CEA testing, cater the type of treatment/resources patients might need in order to complete their visits. They will only be interacting with the interface- they will be inputting the necessary information needed to provide the prediction.  They will have limited technical (computing and modeling) expertise, therefore, the interface needs to be user-friendly. 


## Use cases/Functional Design

### Policy analyst/Physicians
**What information does the user provide?** Provide value for needed features  
**What responses do the system provide?** Getting the likelihood of colorectal patients visiting clinics/CEA testing 


### Researchers
**What information does the user provide?** Their specific dataset  
**What responses do the system provide?** Based on the data, predicting the best model to use 




