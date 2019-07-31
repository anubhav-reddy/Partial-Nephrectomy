
# coding: utf-8

# In[226]:


import pandas as pd
import pickle
import sklearn
df = pd.read_csv("results_shortlist.csv")
def event_handler(df):
    
    df1 = df[0:1].copy()
    df1 = df1[['PROC NAME','GENDER','AGE AT SURGERY','BMI','CLINICAL SIZE (mm)',
     'SYMPTOMS','SOLITARY KIDNEY','SIDE OF SURGERY','FACE','TUMOR lOCATION','PRE-OP HB','PRE-OP HT','PRE-OP CREAT','PRE-OP EGFR',
     'POLAR LOCATION','EXOPHYTIC RATE','CLINICAL SIZE GROUP','CT','CN','CM','RADIUS (maximal diameter in cm)',
     'EXOPHYTIC/ENDOPHYTIC PROPERTIES','ANTERIOR OR POSTERIOR','ASA SCORE','NO OF LESIONS','ACCESS']].copy()

    category_list = ['PROC NAME','GENDER','SYMPTOMS','SOLITARY KIDNEY','SIDE OF SURGERY','FACE','TUMOR lOCATION','POLAR LOCATION',
     'EXOPHYTIC RATE','CLINICAL SIZE GROUP','CT','CN','CM','RADIUS (maximal diameter in cm)','EXOPHYTIC/ENDOPHYTIC PROPERTIES',
     'ANTERIOR OR POSTERIOR','ASA SCORE','ACCESS']

    for i in category_list:
        df1[i] = df1[i].astype('category')

    sample = pd.read_csv('column_list.csv')

    test = pd.concat([sample,pd.get_dummies(df1)],sort = False).fillna(0)

    #loading a model from a file called model.pkl
    stdc = pickle.load(open("stdc.pkl","rb"))

    test = stdc.transform(test)

    #loading a model from a file called model.pkl
    transformer_kpca = pickle.load(open("transformer_kpca.pkl","rb"))

    test = transformer_kpca.transform(test)

    #loading a model from a file called model.pkl
    model = pickle.load(open("logistic_intra_op.pkl","rb"))

    results = model.predict_proba(test)
    prob_no_complications,prob_complications = round(results[0][0],2),round(results[0][1],2)
    return prob_complications

event_handler(df)

