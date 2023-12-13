# ML4CEATesting
Explore patient and physician characteristics that provide insights into the likelihood of a patient undergoing testing in the next three months.
ML4CEATesting is an interactive software system designed to help clinicians identify patients for carcinoembryonic antigen (CEA) testing using a machine learning model.
Use the URL to access the interactive website to input values to get prediction: [https://ceatestingmodel.streamlit.app/]


Installation
------------
To install `ml4cea` you will need to begin by cloning `ml4cea` on your own computer by using the following `git` command:

```
git clone https://github.com/ML4CEA/CEATestingModel.git
```

Next, to install the package, run the `setup.py` file:

```
python setup.py install
```

To ensure that the dependencies to run `ml4cea` are installed on your computer you will want to run the following command:

```
pip install -r requirements.txt
```

You should now be ready to import and use `ml4cea` on your computer.

Examples
---------------------------
To understand how to use ml4cea, please refer to 
the [examples](https://github.com/ML4CEA/CEATestingModel/tree/main/examples) section of this GitHub page where you can find 
examples for doing the following:

- How to generate the trained model based on the user input data
- How to use the interactive website to predict a patient's likelihood of being tested within the next three months
- How to update the trained model based on user input and use the interactive website to predict a patient's likelihood of being tested within the next three months

Tests
---------------------------
Run the command in the root directory:
```
python -m unittest discover -s tests
```

Collborators:
---------------------------

Ron Dickerson <br>
Mahima Joshi <br>
Janet Pan <br>
Anuradha Ramachandran <br>

Contact
---------------------------


