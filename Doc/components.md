## Big component
**Name**: Userinterface\
**What it does**: This is the interface that interacts with the user (Clinician/ Physician) to get the input values of the features for the prediction model\
**Inputs**: Feature values\
##
**Outputs**: Likelihood (prediction outcomes)

## Subcomponent 1
**Name**: AuthenticateInputValues\
**What it does**: Authenticate the feature input keyed in by the user and prompt the user to input the feature information in a certain format if incorrect\
**Inputs**: Feature values\ 
##
**Outputs**: Boolean and if output is false send an alert to the user to re-enter values
 
## Subcomponent 2:
**Name**: PredictionBlock\
**What it does**: Provides prediction for the given values by the user input using the best model selected from the training and validation step.\
**Inputs**: Feature values in the correct format\ 
##
**Output**: Predicted CEA test value obtained from the model.


## Big Component:
**Name**: Model training and validation\
**What it does**: 5 models are trained on the clean dataset and the performance of the model is evaluated on the validation dataset to ascertain the model with the best performance and preserve the best model with its feature weights for future predictions.\
**Inputs**: Electronic records data frame\ 
##
**Output**: Pickle object that contains the feature weights of the robust model 

## Subcomponent 1: 
**Name**: TrainModel\
**What it does**: Train 5 models such as XGBoost, Random Forest, Lasso, Elastic Net, Ridge on the train dataset\
**Inputs**: Training dataset that is a data frame consisting of individual patient records with patient and physician characteristics\
##
**Outputs**: Trained model with feature weights

## Subcomponent 2:
**Name**: Evaluation\
**What it does**: Evaluate the 5 models on the validation set and compare their AUC to get a robust model\
**Inputs**: Model features from the training module and validation data frame\
##
**Outputs**: AUC metric for each model and save only the robust model

## Big Component
**Name**: Data processing\ 
**What it does**: Cleaning the data frame in order to train and validate five different models.\
**Inputs**: Raw data frame\ 
##
**Output**: Clean data frame

