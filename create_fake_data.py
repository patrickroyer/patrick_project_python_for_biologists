import argparse
import string
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser(description = "Create a fake training set, a fake testing set and 5 patients to evaluate")
    parser.add_argument("filename1", metavar = "filename1", type = str, help = "Enter the output file name for training set")
    parser.add_argument("filename2", metavar = "filename2", type = str, help = "Enter the output file name for testing set")
    parser.add_argument("number_of_samples", metavar = "number_of_samples", type = int, help = "Enter the number of samples")
    args = parser.parse_args()

    n_samples = args.number_of_samples
    the_filename_train = args.filename1
    the_filename_test = args.filename2
    
    # Training set: n_samples samples and 100 proteins
    random.seed(1981)
    #print(f"- Create a fake protein matrix with random values [{n_samples} samples, 100 fake proteins]")
    df_train = pd.DataFrame(np.random.random(size=(n_samples, 100)))

    # 50 proteins are related to the outcome y
    relation = [] 
    df_train['y'] = 0

    random.seed(1981)
    #print(f"- Create fake outcome 'mortality' related to the first 50 fake proteins")
    for i in range(0,50):
        temp = random.randint(-10, 10)
        relation.append(temp)
        df_train['y'] += df_train[i]*temp

    # Scale outcome between 0 and 1 to mimic probability of death
    #print(f"- Scale the outcome and make it binary")
    df_train['y'] = (df_train['y'] - df_train['y'].min()) / (df_train['y'].max() - df_train['y'].min())  

    # Convert to binary outcome: probability >= 0.5, y = 1, otherwise y = 0

    df_train['y'] = df_train['y'].apply(lambda x: 0 if x < 0.5 else 1)

    # Add increasing noise to the 50 related proteins
    #print(f"- Add some noise to protein data")
    random.seed(1981)
    for i in range(0,50):
        if i>=0 and i<10:
            df_train[i] += random.uniform(-1,1)*0.001
        if i>=10 and i<20:
            df_train[i] += random.uniform(-1,1)*0.01
        if i>=20 and i<30:
            df_train[i] += random.uniform(-1,1)*0.1
        if i>=30 and i<40:
            df_train[i] += random.uniform(-1,1)*0.2
        if i>=40 and i<50:
            df_train[i] += random.uniform(-1,1)*0.3

    # Min-Max Scaling
    #print(f"- Scale the protein values")
    scaler = MinMaxScaler()    
    df_train.iloc[:,0:100] = scaler.fit_transform(df_train.iloc[:,0:100])

    # Generate fake protein names
    #print(f"- Generate fake protein names")
    head = []
    random.seed(1981)
    for i in range(0,100):
        head.append(''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))
    head.append('mortality')

    df_train.columns = head

    # Export the training set as CSV
    #print(f"- Export the training set as a csv file: {the_filename_train}.csv")
    df_train.to_csv(the_filename_train + ".csv", index = False)

    print("### FAKE TRAINING SET CREATED ############################")
    
    # Take a subset of the trainig set and add some gaussian noise
    #print(f"- Take a subset (n = {round(n_samples/5)}) of the trainig set and add some gaussian noise")
    random.seed(1981)
    df_test = df_train.iloc[random.sample(range(0, n_samples), round(n_samples/5)),:]
    df_test.reset_index(inplace = True, drop = True)

    random.seed(1981)
    noise = np.random.normal(0, 0.3, [df_test.shape[0],df_test.shape[1]-1]) 
    df_test[head[0:100]] = df_test[head[0:100]] + noise

    # Export the testing set as CSV
    #print(f"- Export the testing set as a csv file: {the_filename_test}.csv")
    df_test.to_csv(the_filename_test + ".csv", index = False)

    print("### FAKE TESTING SET CREATED ############################")
    
    # Take a subset of the trainig set and add some gaussian noise 
    random.seed(1982)
    #print("- Take a subset of the trainig set (n = 5) and add some gaussian noise")    
    df_samples = df_train.iloc[random.sample(range(0,n_samples), 5),0:100]
    df_samples.reset_index(inplace = True, drop = True)
    
    random.seed(1982)
    noise = np.random.normal(0, 0.4, [df_samples.shape[0],df_samples.shape[1]]) 
    df_samples = df_samples + noise

    df_sample_1 = df_samples.iloc[[0]]
    df_sample_2 = df_samples.iloc[[1]]
    df_sample_3 = df_samples.iloc[[2]]
    df_sample_4 = df_samples.iloc[[3]]
    df_sample_5 = df_samples.iloc[[4]]
    
    # Export the training set as CSV file
    #print("- Export fake patients to be evaluated as CSV files, patient_1.csv, patient_2.csv, patient_3.csv, patient_4.csv, patient_5.csv") 
    df_sample_1.to_csv('patient_1.csv', index = False)
    df_sample_2.to_csv('patient_2.csv', index = False)
    df_sample_3.to_csv('patient_3.csv', index = False)
    df_sample_4.to_csv('patient_4.csv', index = False)
    df_sample_5.to_csv('patient_5.csv', index = False)

    print("### 5 FAKE PATIENTS CREATED ############################")
    
    print(f"--> Done!")

if __name__ == "__main__":
    main()