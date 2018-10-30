import pandas as pd
import pickle
import sklearn
def event_handler(test):
    test_df = pd.DataFrame(data=test,index=[0])
    model = pickle.load(open("H:\RediMinds\DRMahen/model_intra_op.pkl","rb"))
    prediction = model.predict_proba(test_df)
    prob_no_intra_complications = prediction[0][0]
    prob_intra_complications = prediction[0][1]
    return prob_no_intra_complications, prob_intra_complications