patrick_project_python_for_biologists

# Predicting 10-year risk of cardiovascular disease mortality from plasma proteomics

## Script create_fake_data.py

Create a fake training data set including 100 fake proteins with their plasma values and the mortality outcome (1 csv file)

Create a fake testing data set including the same 100 fake proteins with their plasma values and the mortality outcome (1 csv file)

Create 5 fake patients including the same 100 fake proteins with their plasma values and the mortality outcome (5 csv files)

### Usage: 
    python create_fake_data.py training_set test_set number_of_samples

training_set = name of the created csv training file, output training_set.csv

test_set = name of the created csv test file, output test_set.csv

number_of_samples = number of samples in the created training set

The script will also output 5 fake patients: patient_1.csv, patient_2.csv, patient_3.csv, patient_4.csv and patient_5.csv.

## Script test_fake_data.py

Train a model (xgboost classifier, default parameters) for predicting mortality (on 80% of the training set) and output the accuracy of the validation set (20% of the training set) and the accuracy of the test set.

The script will also output the probability of death for a specified patient

### Usage: 
    python test_fake_data.py training_set.csv test_set.csv patient_1.csv


### Create the environment locally from the environment.yml file 
    conda env create -f environment.yml

