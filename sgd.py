import numpy as np
import random
import csv
import math

#First we need to extract the data from the csv file so we can train and validate our model.
def extract_data():

    #features to hold the data and results to hold the actual chances of admit
    features = [];
    admission = []
    count = 0;
    #we need to get the features and divide them by the total scores you can get and hold them in out featurs vector
    #and hold the chance of admit in the admission vector
    with open('Admission_predict_Ver1.1.csv', 'r') as csv_data:
        r = csv.DictReader(csv_data)
        for row in r:
            #features holds the feature sets for each admission application
            data = np.array([float(row['GRE Score'])/340, float(row['TOEFL Score'])/120, float(row['University Rating'])/5, float(row['SOP'])/5, float(row['LOR '])/5, float(row['CGPA'])/10, float(row['Research']),1]);
            features.append(data)
            #admission holds the chance of admission for each feature set with the same index in features vector
            coa = float(row['Chance of Admit ']);
            admission.append(coa);
    return(features, admission)

