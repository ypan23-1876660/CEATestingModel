# ML4CEATesting
Identify patient and physician characteristics that predict a patient's likelihood of being tested in the next three months

**Collborators:** <br>
Echo Ron <br>
Mahima <br>
Anuradha <br>
Janet Pan <br>


 # How to use
 
 1. Download all the files into the local `/data` folder: <br>
 **Do not push the `/data` folder to Github!**
    - Pateint CEA testing: data/deid_cea_v2.csv
    - Patient information: Final dataset prep_072521.csv
    - Physician data: died_md_specialty_v1.csv, deid_md_dep_v1.csv, deid_md_edu_v1.csv, deid_md_main_v1.csv
    
   
 2. Run `python main.py` to generate the dataframe `patient_phys_info.csv` that includes all the patient information, new patient variables, and physician characteristics. 


 
