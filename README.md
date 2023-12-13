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
--------

To understand how to use ml4cea, please refer to
the [examples](https://github.com/ML4CEA/CEATestingModel/tree/main/examples) section of this GitHub page where you can find
examples for doing the following:

- How to generate the trained model based on the user input data
- How to use the interactive website to predict a patient's likelihood of being tested within the next three months
- How to update the trained model based on user input and use the interactive website to predict a patient's likelihood of being tested within the next three months

Tests
-----

Run the command in the root directory:

```
python -m unittest discover -s tests
```

Folder structure
----------------

Repository is structered as below. 

Data merging and cleaning can be found under ml4cea/create_variable.py and ml4cea/data_clean.py.

Model can be found under ml4cea/model.py.

Tests can be found under tests folder.

Examples for using the prediction portal can be found at examples/ folder

```
.
├── LICENSE
├── README.md
├── data
│   ├── Final dataset prep_072521.csv
│   ├── default_output
│   │   ├── max_train.npy
│   │   ├── min_train.npy
│   │   └── model.pkl
│   ├── deid_cea_v2.csv
│   ├── deid_md_dep_v1.csv
│   ├── deid_md_edu_v1.csv
│   ├── deid_md_main_v1.csv
│   ├── deid_md_specialty_v1.csv
│   ├── modeltestfail.csv
│   ├── modeltestvalid.csv
│   └── modeltestvalid2.csv
├── Doc
│   ├── components.md
│   ├── technology_review
│   │   └── ML4CEA Technology review.pptx
│   └── userStories.md
├── examples
│   ├── example_model_predict.png
│   ├── model_predict.md
│   └── model_train.md
├── ml4cea
│   ├── __init__.py
│   ├── create_variable.py
│   ├── data_clean.py
│   ├── model.py
├── model_predict.py
├── requirements.txt
├── scripts
│   └── model_train.py
├── setup.py
└── tests
    ├── test_create_variable.py
    ├── test_data_clean.py
    └── test_model.py
```


Collborators:
-------------

Ron Dickerson `<br>`
Mahima Joshi `<br>`
Janet Pan `<br>`
Anuradha Ramachandran `<br>`

Contact
-------
