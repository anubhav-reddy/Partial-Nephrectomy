### Test Script

# import packages
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from sklearn.pipeline import Pipeline
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_path = 'output/models/'

list_columns = [
    'CENTERCODE',
    'GENDER',
    'AGEATSURGERY',
    'MARITALSTATUS',
    'RACE',
    'EDUCATION',
    'BMI',
    'CLINICALSIZEmm',
    'CHARLSONSCORE',
    'SYMPTOMS',
    'SOLITARYKIDNEY',
    'BILATERALITYOFTUMOR',
    'SIDEOFTUMOR',
    'SIDEOFSURGERY',
    'FACE',
    'TUMORlOCATION',
    'PREOPHB',
    'PREOPHT',
    'PREOPWBC',
    'PREOPCREAT',
    'PADUARISK',
    'POLARLOCATION',
    'RIMLOCATION',
    'RENALSINUS',
    'EXOPHYTICRATE',
    'CLINICALSIZEGROUP',
    'CT',
    'CN',
    'R.E.N.A.L.NEPHRORISKSTRATIFICATION',
    'RADIUSmaximaldiameterincm',
    'NEARNESSOFTUMOUR',
    'ANTERIORORPOSTERIOR',
    'LOCATIONTOPOLARLINE',
    'ASASCORE',
    'PARTIALNEPHROINDICATION',
    'MULTIFOCALITY',
    'NOOFLESIONS',
    'PATIENTNUMBER'
               ]

# Import prospective dataset
df= pd.read_excel('input/test_data.xlsx', 
                  sheet_name='test_data', 
                  usecols=list_columns)

### Define numeric columns and replace encoded missing values with NaN

# List of columns tobre converted to numeric
numeric_col_list = ['AGEATSURGERY','BMI','CLINICALSIZEmm','CHARLSONSCORE','PREOPHB',
                    'PREOPHT','PREOPWBC','PREOPCREAT','NOOFLESIONS'
                   ] 

# Convert columns in numeric_col_list to numeric and invalid values are set NaN 
for col in numeric_col_list:
    df[col]= pd.to_numeric(df[col], errors='coerce', downcast = 'float')

# chekc if any column is still integer an dconvert it to float
for i in df.select_dtypes(include = 'int64').columns:
    df[i] = df[i].astype('float')

#replace missing values such as 999 in the dataframe with NaN
df = df.replace([99,999,9999,99999,999999,-99,-999,-9999,-99999,-999999],np.nan)

# replacing negative numbers in the dataframe with nan as given variables cannot contain negative numbers
for col in list(df.select_dtypes('float64')):
    df[col] = df[col].apply(lambda x: np.nan if x<0 else x)

# Correcting the units for erroroneously entered data
def clean_WBC(x):
    if len(str(x))<6:
        x = x*1000
    return x
        

# if the value of PRE-OP WBC value contains is less the 4 digits then multiply it by 1000
df['PREOPWBC'] = df['PREOPWBC'].apply(lambda x: clean_WBC(x))


# Correcting the units for erroroneously entered data for PREOPHB
df['PREOPHB'] = df['PREOPHB'].apply(lambda x: x*100 if x<10 else x)

# Correcting the units for erroroneously entered data for PREOPHT
df['PREOPHT'] = df['PREOPHT'].apply(lambda x: x*100 if x<10 else x)

# define categorical columns
cat_col = []
for i in list_columns:
    if i not in numeric_col_list:
        cat_col.append(i)
cat_col.remove('PATIENTNUMBER')

### Data Cleaning

import json

with open(model_path + "outlier_dict.json", "r") as read_file:
    outlier_dict = json.load(read_file)

outlier_dict

df['PREOPWBC']

# Remove outlier data from numeric columns
for i in numeric_col_list:
    if i in outlier_dict:
        LL = outlier_dict[i]['LL']
        UL = outlier_dict[i]['UL']
        df.drop(df.loc[(df[i]<=LL)|(df[i]>=UL),[i]].index, inplace=True)

df

# Calculate number missing values per row
df.reset_index(inplace=True, drop=True)
missing = {}

for i in range(len(df)):
    miss_cnt = 0
    for col in df.columns:
        if pd.isna(df[col][i]) == True:
            miss_cnt = miss_cnt+1
    df.loc[i,'Missing'] = miss_cnt

# calculate number of records with over 30% missing data
print("Total Records {}".format(len(df)))
print("Records with >=30% missing data {}".format(sum(df['Missing']<= round((len(df.columns))*.30))))
print("Records to be dropped {}".format(len(df) - sum(df['Missing']<= round((len(df.columns))*.30))))

#removing patients with more than 75% missing data
df = df[df['Missing']<= round((len(df.columns))*.30) ].copy()

# remove column 'Missing'
df.drop(labels=['Missing'], axis = 'columns',inplace=True)

# Replcaing missing values in categorical column with NA
for k in cat_col:
    if k in df.columns:
        df[k].fillna('NA',inplace = True)

# Import dict with mean value for numeric columns
with open(model_path + "numeric_col_mean_dict.json", "r") as read_file:
    numeric_col_mean_dict = json.load(read_file)

# Replacing missing values in numerical columns with their respective mean 
for k in numeric_col_list:
    if k in df.columns:
        df[k].fillna(numeric_col_mean_dict[k], inplace = True)

from joblib import load
le_dict = load(model_path+'Label_enc_dict.joblib')

# Convert all cateogir
for i in cat_col:
    df[i] = df[i].apply(str)

# check for unknow labels and map unknown labels to 'unknown_label'
for i in le_dict:
    if i in df.columns:
        df[i] = df[i].map(lambda x: x if x in le_dict[i].classes_ else 'unknown_label').copy()

df_codes = df.copy()

# mapping categorical column values to integer labels ##Still need to work on unknown_label
for col in cat_col:
    print(col)
    df_codes[col] = le_dict[col].transform(df_codes[col]).copy()

from joblib import load
patient_list_train = load(model_path+'patient_list_test.joblib')

x_test = df_codes[df_codes['PATIENTNUMBER'].isin(patient_list_train)]

patient_number = x_test['PATIENTNUMBER'].to_list()
x_test.drop(labels= 'PATIENTNUMBER', axis = 'columns', inplace = True)
x_test.reset_index(inplace=True, drop = True)

### Prediction

# Generate prediction for the Random Forest Model
with open (model_path +'\cat_col', 'rb') as fp:
    cat_col = pickle.load(fp)

model_path = 'output/models/'
from joblib import load
encoder = load(model_path+'OHE.joblib')

#x_test = test.drop(labels='INTRA_OP_COMPLICATIONS', axis = 'columns').copy()
#y_test = test['INTRA_OP_COMPLICATIONS'].copy() 

# Create dummy variables
one_hot_encoded_array = encoder.transform(x_test[cat_col]).toarray()
column_name = encoder.get_feature_names(cat_col)
x_test_OHE =  pd.DataFrame(one_hot_encoded_array, columns= column_name)
x_test = x_test.merge(x_test_OHE, how = 'left', left_index = True, right_index =True) # create dummy variables
x_test = x_test.drop(labels = cat_col, axis = 'columns') # drop original variables

### Load Models

model_path = 'output/models/'

# import Random Forest Classifer
from joblib import load
RFR = load(model_path+'RFR.joblib')

# Generate prediction for the Random Forest Model
results_RFR = pd.DataFrame(RFR.predict(x_test), columns=['pred_label'])
results_RFR['pred_prob'] =  pd.DataFrame(RFR.predict_proba(x_test))[1]
results_RFR['PATIENTNUMBER'] = patient_number
#results_RFR['true_label'] = np.array(y_test)

print(results_RFR[['PATIENTNUMBER','pred_prob','pred_label']])