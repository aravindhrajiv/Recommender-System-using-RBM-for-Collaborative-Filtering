
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

import torch

from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# In[44]:


unique_var = ['Customer_ID']
cont_var = ['Age','Relnshp_Mnths','Gross_household_income']
cate_var = ['Employee_Index','Sex','New_customer','Relnshp_flag','Cust_type_beg_Mth','Cust_Reln_type_beg_mth','Residence_flag',
            'Foreigner_flag','Emp_spouse_flag','Deceased_flag','Address_type','Activity_flag','Segment']
cate_var_date = ['Month_status_date','Join_date','Last_date_Prim_Cust']
cate_var_big = ['Customer_country','Channel_when_joined','Customer_address','Address_detail']
cate_var_int = ['New_customer','Relnshp_flag','Activity_flag','Address_type']
target_var = ['Saving_account','Guarantees','Cur_account','Derivative_account','Payroll_account','Junior_account','Particular_acct1',
              'Particular_acct2','Particular_acct3','Short_term_deposites','Med_term_deposites','Long_term_deposites','e-account', 
              'Funds','Mortgage','Pension','Loans','Taxes','Credit_card','Securities','Home_account','Payroll','Pensions','Direct_debit']
str_ = ['Employee_Index','Customer_country','Sex','New_customer','Relnshp_flag','Cust_type_beg_Mth','Cust_Reln_type_beg_mth',
            'Residence_flag','Foreigner_flag', 'Emp_spouse_flag', 'Channel_when_joined','Address_type','Deceased_flag','Customer_address',
            'Address_detail','Activity_flag','Segment']
tar_var_sp = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
              'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
              'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
              'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

# In[50]:


class RBM():
    def __init__(self, nv, nh, r):
        self.W = torch.randn(r, nh, nv)*0.1
        self.a = torch.randn(1,1,nh)*0.01
        self.b = torch.randn(1,1,nv)*0.01
        
    def sample_h(self,x):
        wx = torch.matmul(x.permute(2,0,1), self.W.permute(0,2,1))
        activation = wx + self.a.expand_as(wx)
        p = torch.sigmoid(activation)
        return p, p.bernoulli()

    def sample_v(self,y):
        wy = torch.matmul(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p = torch.sigmoid(activation.permute(1,2,0))
        return p, p.bernoulli()
    
    def train_rbm_fn(self,v0,vk,ph0,phk,batch_size,learning_rate = 0.0000001):
        self.W += learning_rate * (torch.matmul(v0.permute(2,1,0),ph0) - torch.matmul(vk.permute(2,1,0), phk)).permute(0,2,1)/ batch_size
        self.b += learning_rate * torch.sum(torch.sum((v0-vk).permute(2,0,1),1),0)/ batch_size
        self.a += learning_rate * torch.sum(torch.sum((ph0-phk),1),0)/ batch_size
    
    def free_energy(self, v):
        v_term = -torch.sum(torch.matmul(v.permute(2,0,1), self.b.permute(0,2,1)),dim=1)
        wx = torch.matmul(v.permute(2,0,1), self.W.permute(0,2,1))
        w_x_h = wx + self.a.expand_as(wx)
        h_term = -torch.sum(torch.sum(F.softplus(w_x_h),dim=2),dim=1)
        fe = torch.mean((h_term+v_term))
        return fe


dtype_list = {'fecha_dato': 'object', 'ncodpers': 'int64', 'ind_empleado': 'str','pais_residencia':'str', 'sexo':'str', 'age': 'int64', 'fecha_alta': 'object',
                'ind_nuevo': 'str', 'antiguedad':'int64', 'indrel': 'int32','ult_fec_cli_1t': 'object','indrel_1mes': 'int32', 'tiprel_1mes':'str','indresi': 'str',
                'indext': 'str','conyuemp': 'str', 'canal_entrada':'str','indfall': 'str','tipodom':'int32', 'cod_prov':'int64','nomprov': 'str','ind_actividad_cliente': 'int32',
                'renta': 'float64', 'segmento' :'str','ind_cco_fin_ult1': 'int16', 'ind_deme_fin_ult1': 'int16', 'ind_aval_fin_ult1': 'int16', 'ind_valo_fin_ult1': 'int16', 
                'ind_reca_fin_ult1': 'int16', 'ind_ctju_fin_ult1': 'int16','ind_cder_fin_ult1': 'int16', 'ind_plan_fin_ult1': 'int16', 'ind_fond_fin_ult1': 'int16',
                'ind_hip_fin_ult1': 'int16', 'ind_pres_fin_ult1': 'int16', 'ind_nomina_ult1': 'int16', 'ind_cno_fin_ult1': 'int16','ind_ctpp_fin_ult1': 'int16', 
                'ind_ahor_fin_ult1': 'int16', 'ind_dela_fin_ult1': 'int16', 'ind_ecue_fin_ult1': 'int16', 'ind_nom_pens_ult1': 'int16', 'ind_recibo_ult1': 'int16', 
                'ind_deco_fin_ult1': 'int16', 'ind_tjcr_fin_ult1': 'int16', 'ind_ctop_fin_ult1': 'int16', 'ind_viv_fin_ult1': 'int16', 'ind_ctma_fin_ult1': 'int16'}

train_0 = pd.read_csv("Dataset/reference_sample_001.csv",header = None,dtype = dtype_list,na_values = 'NA')
train_0.columns = ['Month_status_date', 'Customer_ID', 'Employee_Index', 'Customer_country', 'Sex', 'Age', 'Join_date',
                'New_customer', 'Relnshp_Mnths', 'Relnshp_flag','Last_date_Prim_Cust', 'Cust_type_beg_Mth', 'Cust_Reln_type_beg_mth',
                'Residence_flag', 'Foreigner_flag', 'Emp_spouse_flag', 'Channel_when_joined', 'Deceased_flag', 
                'Address_type', 'Customer_address', 'Address_detail', 'Activity_flag', 'Gross_household_income',
                'Segment', 'Saving_account', 'Guarantees', 'Cur_account', 'Derivative_account', 'Payroll_account',
                'Junior_account', 'Particular_acct1', 'Particular_acct2', 'Particular_acct3', 'Short_term_deposites',
                'Med_term_deposites', 'Long_term_deposites', 'e-account', 'Funds', 'Mortgage', 'Pension', 'Loans',
                'Taxes', 'Credit_card', 'Securities', 'Home_account', 'Payroll', 'Pensions', 'Direct_debit']

train_1= train_0[1:]
# In[112]:


def user_data(input_0,clf,rbm,k):
    
    input_0.columns = ['Month_status_date', 'Customer_ID', 'Employee_Index', 'Customer_country', 'Sex', 'Age', 'Join_date',
                'New_customer', 'Relnshp_Mnths', 'Relnshp_flag','Last_date_Prim_Cust', 'Cust_type_beg_Mth', 'Cust_Reln_type_beg_mth',
                'Residence_flag', 'Foreigner_flag', 'Emp_spouse_flag', 'Channel_when_joined', 'Deceased_flag', 
                'Address_type', 'Customer_address', 'Address_detail', 'Activity_flag', 'Gross_household_income','Segment']

    if(int(input_0['Customer_ID'].values.item()) in train_1['Customer_ID'].values):
        a = train_1[train_1['Customer_ID'].values == int(input_0['Customer_ID'].values.item())]
        print('Old User')
        recommendation = old_user(a[a.columns[24:]].iloc[0,:],input_0,rbm,k)
    else:
        a = pp(input_0)
        print('New User')
        recommendation = new_user(a,input_0,clf,rbm,k)
    
    return recommendation


# In[47]:


def pp(test_1):
    
    for ind_, var_ in enumerate(cont_var):
        test_1[var_] = pd.to_numeric(test_1[var_],errors='coerce')
    for ind, var in enumerate(cate_var_int): 
        test_1[var] = test_1[var].astype(float).astype(int)
    
    # Missing values for continuous variable - Train
    for ind,var in enumerate(cont_var):
        test_1[var] = test_1[var].replace(to_replace = ['NA',' NA','         NA',-999999], value = np.nan)
        test_1[var].fillna(pd.to_numeric(test_1[var],errors='coerce').dropna().mean(),inplace=True)
    
    # Missing values for categorical variable - Train
    for ind,var in enumerate(cate_var):
        test_1[var] = test_1[var].replace(to_replace = ['NA',' NA','     NA'], value = np.nan)
        test_1[var].fillna(test_1[var].value_counts().index[0],inplace=True)
    for ind,var in enumerate(cate_var_big):
        test_1[var] = test_1[var].replace(to_replace = ['NA',' NA','     NA'], value = np.nan)
        test_1[var].fillna(test_1[var].value_counts().index[0],inplace=True)
    for ind,var in enumerate(cate_var_date):
        test_1[var] = test_1[var].replace(to_replace = ['NA',' NA','     NA'], value = np.nan)
        test_1[var].fillna(test_1[var].value_counts().index[0],inplace=True)
    
    for ind, var in enumerate(cate_var_int): 
        test_1[var] = test_1[var].astype(float).astype(int)
    
    # string to categorical
    test_1[str_] = test_1[str_].astype('category')
    # Dropping Data columns
    test_1.drop(columns=cate_var_date,inplace=True)
    
    le = preprocessing.OneHotEncoder(sparse=False)
    for column_name in str_:
        test_1[column_name] = le.fit_transform(np.array(test_1[column_name].astype(str)).reshape(-1,1)).astype(float)
    
    return test_1    


# In[54]:


def old_user(t1,input_0,rbm,k):
    
    test_rbm = np.array(t1)
    test_rbm = np.reshape(test_rbm, (-1,24,1))
    test_rbm = torch.FloatTensor(test_rbm)
    
    rbm_preds = rbm_model(rbm,test_rbm)
    final_preds = predict(rbm_preds,input_0,k)
    
    return final_preds


# In[55]:


def new_user(test_rf,input_0,clf,rbm,k):
    
    # Testing Random Forest Classifiers for each product
    df_test = []
    for ind, var in enumerate(target_var):
        df_test.append((clf[ind].predict_proba(test_rf)))  
        
    a = np.array(df_test).T
    t1 = pd.DataFrame(a[1],columns=target_var)
    t1 = pd.DataFrame(np.where(t1.T == t1.T.max(), 1, -1),index=t1.columns).T
    test_rbm = np.array(t1)
    test_rbm = np.reshape(test_rbm, (-1,24,1))
    test_rbm = torch.FloatTensor(test_rbm)
    
    rbm_preds = rbm_model(rbm,test_rbm)
    final_preds = predict(rbm_preds,input_0,k)
    
    return final_preds
# In[56]:


def rbm_model(rbm,test_rbm):
    nv = 24
    nh = 1000
    r = 1   # No. of star ratings
    k1 = 1

    vk = test_rbm
    for k in range(k1):
        _,hk = rbm.sample_h(vk)
        vk_orig,vk = rbm.sample_v(hk)
    phk,_ = rbm.sample_h(vk)
    rbm_preds = vk_orig.numpy()
    
    return rbm_preds


# In[114]:


def predict(rbm_preds,input_0,k=3):
    
    a = np.array(np.reshape(rbm_preds,(-1,24)))
    a = pd.DataFrame(a,columns=tar_var_sp)
    t = input_0['Customer_ID']
    pred = pd.concat([t, a],axis=1)
    pred.columns = ['Customer_ID','Saving_account','Guarantees','Cur_account','Derivative_account','Payroll_account','Junior_account','Particular_acct1',
              'Particular_acct2','Particular_acct3','Short_term_deposites','Med_term_deposites','Long_term_deposites','e-account', 
              'Funds','Mortgage','Pension','Loans','Taxes','Credit_card','Securities','Home_account','Payroll','Pensions','Direct_debit']
    o = pred[pred.columns[1:]].sort_values(pred[pred.columns[1:]].last_valid_index(),axis=1, ascending=False, inplace=False, kind='quicksort', na_position='last').T
    p = pd.DataFrame(pred[pred.columns[0]],columns = ['Customer_ID'])
    q = pd.DataFrame(o.index[0:k],columns = ['Product Recommended'])
    final_pred = pd.concat([p,q],axis=1)
    
    return final_pred