
# coding: utf-8

# In[28]:


import pandas as pd


# In[30]:


test = {"PROC NAME":0,"GENDER":1,"SYMPTOMS":1,"SOLITARY KIDNEY":1,
        "SIDE OF SURGERY":3,"FACE":2,"TUMOR LOCATION":1,"POLAR LOCATION":1,
        "EXOPHYTIC RATE":2,"CLINICAL SIZE GROUP":1,"CT":4,"CN":3,"CM":0,
        "R.E.N.A.L. NEPHRO RISK STRATIFICATION":1,"RADIUS (maximal diameter in cm)":1,
        "EXOPHYTIC/ENDOPHYTIC PROPERTIES":2,"ANTERIOR OR POSTERIOR":2,"ASA SCORE":2,
        "PARTIAL NEPHRO INDICATION":1,"ACCESS":1,"AGE AT SURGERY":23,"BMI":34,"CLINICAL SIZE":2,
        "CHARLSON SCORE":4,"CHARLSON AGE-ADJUST SCORE":4,"PRE-OP HB":4,"PRE-OP HT":4,
        "PRE-OP CREAT":4,"PRE-OP EGFR":4,"NO OF LESIONS":4}


# In[31]:


test_df = pd.DataFrame(data=test,index=[0])


# In[32]:


import pickle
#loading a model from a file called model.pkl
model = pickle.load(open("H:\RediMinds\DRMahen/model_intra_op.pkl","rb"))


# In[33]:


prediction = model.predict_proba(test_df)


# In[35]:


prob_no_intra_complications = prediction[0][0]
prob_intra_complications = prediction[0][1]

