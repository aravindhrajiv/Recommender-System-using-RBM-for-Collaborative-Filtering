from flask import Flask, request, jsonify, render_template,redirect,make_response,url_for,send_from_directory, Response
from werkzeug.utils import secure_filename
import shutil
import json
import re
import requests
import os
import sys
import pandas as pd
from pandas.io.json import json_normalize
from main import *
import numpy as np
import networkx as nx
import ast
from itertools import combinations
import pickle

dtype_input = {'Month_status_date': 'object', 'Customer_ID': 'int64'
               ,'Employee_Index': 'str','Customer_country':'str', 'Sex':'str', 'Age': 'int64', 'Join_date': 'object'
               ,'New_customer': 'str', 'Relnshp_Mnths':'int64', 'Relnshp_flag': 'int32','Last_date_Prim_Cust': 'object','Cust_type_beg_Mth': 'int32', 'Cust_Reln_type_beg_mth':'str','Residence_flag': 'str'
               ,'Foreigner_flag': 'str','Emp_spouse_flag': 'str', 'Channel_when_joined':'str','Deceased_flag': 'str','Address_type':'int32', 'Customer_address':'int64','Address_detail': 'str','Activity_flag': 'int32'
               ,'Gross_household_income': 'float64', 'Segment' :'str'
                }

clf = [None]*len(target_var)
for ind, var in enumerate(target_var):
    # Loading the classifier
    clf[ind] = pickle.load(open('Trained Model/Random Forest_trial/'+str(var), 'rb'))

# load the model from disk
filename = 'Trained Model/RBM/rbm_model.sav'
rbm = pickle.load(open(filename, 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/RecommenderSystem', methods=['POST'])
def RecommenderSystem():

    if request.method == 'POST':

        user_data = { "Month_status_date" : [request.form.get('Month_status_date')],
        'Customer_ID' : [request.form.get('Customer_ID')], 
        'Employee_Index' : [request.form.get('Employee_Index')],
        'Customer_country' : [request.form.get('Customer_country')], 
        'Sex' : [request.form.get('Sex')], 
        'Age' : [request.form.get('Age')], 
        'Join_date' : [request.form.get('Join_date')],
        'New_customer' : [request.form.get('New_customer')], 
        'Relnshp_Mnths' : [request.form.get('Relnshp_Mnths')], 
        'Relnshp_flag' : [request.form.get('Relnshp_flag')],
        'Last_date_Prim_Cust' : [request.form.get('Last_date_Prim_Cust')], 
        'Cust_type_beg_Mth' : [request.form.get('Cust_type_beg_Mth')], 
        'Cust_Reln_type_beg_mth' : [request.form.get('Cust_Reln_type_beg_mth')],
        'Residence_flag' : [request.form.get('Residence_flag')], 
        'Foreigner_flag' : [request.form.get('Foreigner_flag')], 
        'Emp_spouse_flag' : [request.form.get('Emp_spouse_flag')], 
        'Channel_when_joined' : [request.form.get('Channel_when_joined')], 
        'Deceased_flag' : [request.form.get('Deceased_flag')],
        'Address_type' : [request.form.get('Address_type')], 
        'Customer_address' : [request.form.get('Customer_address')], 
        'Address_detail' : [request.form.get('Address_detail')], 
        'Activity_flag' : [request.form.get('Activity_flag')], 
        'Gross_household_income' : [request.form.get('Gross_household_income')],
        'Segment' : [request.form.get('Segment')] }

        k = request.form.get('k')
        df_data = pd.DataFrame(user_data, columns = train_0.columns[0:24])
        #print(df_data)
        df_data.dtype = {'Month_status_date': 'object', 'Customer_ID': 'int64'}
        if(k == ""):
            k = 3
        else:
            k = int(float(k))
        final_result = make_pred(df_data,k)
        customer_id = final_result.iloc[0]['Customer_ID']
        pr_rec = final_result['Product Recommended'].values
    		
    return render_template('result.html',user_data = user_data,customer_id = customer_id,pr_rec = pr_rec)

def make_pred(file,k):
    recommendation = user_data(file,clf,rbm,k)
    return recommendation

if __name__ == '__main__':
    app.run(host="0.0.0.0",port = 6002,threaded=False)
    

