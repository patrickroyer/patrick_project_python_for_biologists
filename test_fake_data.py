import argparse
import string
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():

    parser = argparse.ArgumentParser(description = "Do something")
    parser.add_argument("filename_training_set", metavar = "filename_training_set", type = str, help = "Enter the training set file name")
    parser.add_argument("filename_testing_set", metavar = "filename_testing_set", type = str, help = "Enter the testing set file name")
    parser.add_argument("filename_patient", metavar = "filename_patient", type = str, help = "Enter the patient file name")
    
    args = parser.parse_args()

    the_filename_training_set = args.filename_training_set
    the_filename_test_set = args.filename_testing_set
    the_filename_patient = args.filename_patient

    # Import files
    #print("- Import files")
    df_train = pd.read_csv(the_filename_training_set)
    df_test = pd.read_csv(the_filename_test_set)
    df_sample = pd.read_csv(the_filename_patient)
    
    # Create features X and outcome Y for the training set
    X = df_train.iloc[:,0:100]
    Y = df_train.iloc[:,100]

    #print("- Split the training set in training (80%) and validation (20%) sets")
    # Split in training and validation sets
    seed = 1981
    validation_size = 0.33
    X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # Fit the model
    #print("- Fit the model with the training set")
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Predictions for the validation set
    y_pred_validation = model.predict(X_validation)
    predictions_validation = [round(value) for value in y_pred_validation]

    # Evaluate predictions for the validation set
    #print("- Evaluate the predictions for the validation set")
    accuracy_validation = accuracy_score(y_validation, predictions_validation)
    print(f"Accuracy of the validation set: {round(accuracy_validation* 100.0, 1)} %")

    # Create features X and outcome Y for the test set
    X_test = df_test.iloc[:,0:100]
    y_test = df_test.iloc[:,100]

    # Predictions for the test set
    y_pred_test = model.predict(X_test)
    predictions_test = [round(value) for value in y_pred_test]

    # Evaluate predictions for the testing set
    #print("- Evaluate the predictions for the testing set")
    accuracy_test = accuracy_score(y_test, predictions_test)
    print(f"Accuracy of the testing set: {round(accuracy_test* 100.0, 1)} %")

    #print(f"- Evaluate the predictions for the patient: {the_filename_patient}")
    # Predict mortality for new samples
    print(f"10-year risk of death for {the_filename_patient[0:10]}: {round(model.predict_proba(df_sample)[[0]][0][1]*100,1)} %")



if __name__ == "__main__":
    main()