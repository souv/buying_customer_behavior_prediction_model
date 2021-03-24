#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from scipy.sparse import dok_matrix
from datetime import datetime
from sklearn import metrics,preprocessing
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,precision_recall_curve
from sklearn import tree
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.cluster import KMeans,DBSCAN
import pickle as pkl
import os


# In[2]:


#各路徑處理


# In[3]:


print(os.getcwd())
#direct to 財管可活化流失客模型(交接)的位置
os.chdir('..')
os.chdir('..')

print(os.getcwd())


# In[5]:


#####1.匯入基礎資料#####


# In[6]:


cust_all_2018 = pd.read_csv('建模原始資料\\CUST_ALL2018_PRO_0217.csv',engine = 'python') 

print(cust_all_2018.info())
print(cust_all_2018.head(10))


# In[7]:


#####2.用2017年分群結果建構2018年分群結果


# In[8]:


#2018年客戶RFM的變數用於做客戶分群


# In[9]:


status = ['活躍客','流失客']
cust_all_alive_lose_2018 = cust_all_2018[cust_all_2018.STATUS18_19.isin(status)]

pd.set_option('display.max_columns',None)
print(cust_all_alive_lose_2018.info())
print(cust_all_alive_lose_2018.head(10))

cust_all_var_2018 = cust_all_alive_lose_2018[[
        'R_MF2','R_BD2','R_ETF2','R_INS2','R_SI2',
        'F_MF','F_BD','F_ETF','F_SI','F_INS',
        'M_MF','M_BD','M_ETF','M_SI','M_INS']]

cust_all_var_2018.head(10)


# In[10]:


####資料清洗


# In[11]:


####遺失值處理(F,M的用0填補(NaN->0)、R用999填補(NaN->999))
cust_all_var_r_2018 = cust_all_var_2018[['R_MF2','R_BD2','R_ETF2','R_INS2','R_SI2']]

cust_all_var_r2_2018 = cust_all_var_r_2018.replace(to_replace = np.nan,value = 999)

cust_all_var_fm_2018 = cust_all_var_2018[['F_MF','F_BD','F_ETF','F_SI','F_INS',
                                'M_MF','M_BD','M_ETF','M_SI','M_INS']]

cust_all_var_fm2_2018 = cust_all_var_fm_2018.replace(to_replace = np.nan,value = 0)

cust_all_var2_2018 = pd.concat([cust_all_var_r2_2018,cust_all_var_fm2_2018],axis = 1)

print(cust_all_var2_2018.head(10))

#min_max_scaler標準化
min_max_scaler = preprocessing.MinMaxScaler()
cust_all_var2_scale_2018 = min_max_scaler.fit_transform(cust_all_var2_2018)

cust_all_var2_scale_df_2018 = pd.DataFrame(cust_all_var2_scale_2018)
print(cust_all_var2_scale_df_2018.head(10))


# In[15]:


###透過2017年客戶分群的Kmeans分群中心點來為2018年客戶做分群#


# In[14]:


kmeans_init = np.load('建模中間資料\\初始模型\\kmeans_init.npy')

kmeans_model = KMeans(n_clusters = 9,random_state = 10,init = kmeans_init,n_init = 1).fit(cust_all_var2_scale_df_2018)

cust_all_alive_lose_2018['predict_cluster_kmeans']= kmeans_model.fit_predict(cust_all_var2_scale_df_2018)

#cust_all_var2_2018['predict_cluster_kmeans'] = kmeans_model.fit_predict(cust_all_var2_scale_df_2018)

print(cust_all_alive_lose_2018['predict_cluster_kmeans'].value_counts())

#cust_all_s = cust_all_alive_lose_2018[['ID','ID2017','ID2018','ID2019','ID2020','STATUS18_19',
#                                  'M_MF','M_BD','M_ETF','M_SI','M_INS',
#                                  'R_MF2','R_BD2','R_ETF2','R_SI2','R_INS2',
#                                  'F_MF','F_BD','F_ETF','F_SI','F_INS',
#                                  'predict_cluster_kmeans']]

#kmeans_cluster_describe_mean_k = cust_all_s.groupby(['predict_cluster_kmeans','STATUS18_19']).agg(['mean'],as_index = False)
#print(kmeans_cluster_describe_mean_k)

#kmeans_cluster_describe_mean_k.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V3\\2018年分群預測(用2017分群結果).csv',
#                                     encoding = 'utf_8_sig')


# In[16]:


#####3.流失戶預測模型#####


# In[17]:


status = ['流失客']

cust_all_lose_2018 = cust_all_alive_lose_2018[cust_all_alive_lose_2018.STATUS18_19.isin(status)]

print(cust_all_lose_2018.info())
print(cust_all_lose_2018.head(50))


# In[18]:


#recode the Y
#ID2020:NaN是未再購(recode = 0)、非NaN是再購(recode = 1)

#cust_all_lose_2018['ID2020'] = cust_all_lose['ID2020'].astype(str)

cust_all_lose_2018['BUY_INS2020'] = cust_all_lose_2018['ID_INS2020'].isnull().map({True : 0,False: 1})

print(cust_all_lose_2018['BUY_INS2020'].value_counts())

print(cust_all_lose_2018.head(10))


# In[19]:


#連續型變數資料處理


# In[20]:


#計算客戶目前年齡
pd.set_option('mode.chained_assignment',None)
cust_all_lose_2018['year_old_2020'] = round((pd.to_datetime('2020-12-31') - pd.to_datetime(cust_all_lose_2018['BIRTH_DATE_1'],format = '%Y%m%d'))/np.timedelta64(1,'Y'),0)


# In[21]:


#除RFM之外的連續型變數
cust_all_conts_nrfm = cust_all_lose_2018[['D_PB','D_CD','D_FD','D_FND','MF_境內','MF_境外','D_INS','D_BD','D_SI','D_EFT','WM_DTOTAL',
                                       'D_TOTAL','M_TOTAL','TWD_DEP_RATIO_1812','FOR_DEP_RATIO_1812','INS_RATIO_1812','MF_RATIO_1812',
                                       'BD_RATIO_1812','SI_RATIO_1812','ETF_RATIO_1812','DTTL_1803','DTTL_1806','DTTL_1809','DTTL_1812',
                                       'DTTL_DIFF_180306','DTTL_DIFF_180306_R','DTTL_DIFF_180609','DTTL_DIFF_180609_R',
                                       'DTTL_DIFF_180912','DTTL_DIFF_180912_R','CC_1812','CC_1912','ML_1812','ML_1912',
                                       'PL_1812','PL_1912','RPL_1812','RPL_1912','LUM_1812','LUM_1912','INTER_CNT','WM_BUY_PERIOD',
                                       'AVG_INS','AVG_INS_PERC','AVG_MF','AVG_MF_PERC','year_old_2020']]

print(cust_all_conts_nrfm.head(10))
print(cust_all_conts_nrfm.info())


# In[22]:


#RFM相關的連續型變數
cust_all_conts_rfm = cust_all_lose_2018[['R_MF2','F_MF','M_MF','R_BD2','F_BD','M_BD',
        'R_ETF2','F_ETF','M_ETF','R_SI2','F_SI','M_SI',
        'R_INS2','F_INS','M_INS','R_HTD','F_HTD','M_HTD']]

cust_all_var_r = cust_all_conts_rfm[['R_MF2','R_BD2','R_ETF2','R_INS2','R_SI2','R_HTD']]

cust_all_var_r2 = cust_all_var_r.replace(to_replace = np.nan,value = 999)

cust_all_var_fm = cust_all_conts_rfm[[
                                'F_MF','F_BD','F_ETF','F_SI','F_INS',
                                'M_MF','M_BD','M_ETF','M_SI','M_INS','F_HTD','M_HTD']]

cust_all_var_fm2 = cust_all_var_fm.replace(to_replace = np.nan,value = 0)

cust_all_conts2_rfm = pd.concat([cust_all_var_r2,cust_all_var_fm2],axis = 1)

#合併 RFM 變數及非 RFM 變數
cust_all_conts = pd.concat([cust_all_conts2_rfm,cust_all_conts_nrfm],axis = 1)

#將遺失值轉為0
cust_all_conts = cust_all_conts.replace(to_replace = np.nan,value = 0)

print(cust_all_conts.info())
print(cust_all_conts.head(10))


# In[23]:


#連續型變數標準化(平均數與標準差做標準化)
scaler_standard = preprocessing.StandardScaler()

cust_all_conts_s_2018 = scaler_standard.fit_transform(cust_all_conts)

cust_all_conts_s2_2018  = pd.DataFrame(cust_all_conts_s_2018)

cust_all_conts_s3_2018 = pd.concat([cust_all_conts_s2_2018,cust_all_lose_2018['BUY_INS2020'].reset_index(drop = True)],axis = 1)

print(cust_all_conts_s3_2018.head(10))

#產製變數名稱
column_list = cust_all_conts.columns.tolist()
column_list.append('BUY_INS2020')

cust_all_conts_s3_2018.columns = column_list

cust_all_conts_s3_2018.info()
cust_all_conts_s3_2018.head(10)


# In[24]:


#類別型變數資料處理
cust_all_cate_2018 = cust_all_lose_2018[['SEX_CODE','EDUC_CODE','EMP_CATEGORY',
                               'OCCUPATION_CODE2','MARITAL_STATUS',
                               'predict_cluster_kmeans','SEG2_1812','SEG2_1912']]

cust_all_cate_2018.info()
cust_all_cate_2018.head(10)

#將空格填寫為missing
cust_all_cate2_2018 = cust_all_cate_2018.replace(to_replace = r'^\s*$',value = "missing",regex =True)

#將NaN填寫為missing
cust_all_cate2_2018 = cust_all_cate_2018.replace(to_replace = np.nan,value = "missing",regex =True)

print(cust_all_cate2_2018.head(10))


# In[25]:


print(cust_all_cate2_2018['SEX_CODE'].value_counts())

print(cust_all_cate2_2018['EDUC_CODE'].value_counts())

print(cust_all_cate2_2018['EMP_CATEGORY'].value_counts())

print(cust_all_cate2_2018['OCCUPATION_CODE2'].value_counts())

print(cust_all_cate2_2018['MARITAL_STATUS'].value_counts())

print(cust_all_cate2_2018['predict_cluster_kmeans'].value_counts())

print(cust_all_cate2_2018['SEG2_1812'].value_counts())

print(cust_all_cate2_2018['SEG2_1912'].value_counts())


# In[26]:


#類別型 one hot encoding

#load enc pkl 檔 
filename = '建模中間資料\\優化模型\\enc_ins_2.pkl'

enc = pkl.load(open(filename,'rb'))

ohe_cate_data_2018 = enc.transform(cust_all_cate2_2018).toarray()

feature_names = enc.get_feature_names(['SEX_CODE','EDUC_CODE','EMP_CATEGORY',
                              'OCCUPATION_CODE2','MARITAL_STATUS','predict_cluster_kmeans','SEG2_1812','SEG2_1912'])

cust_all_cate_ohe_2018 = pd.DataFrame(ohe_cate_data_2018,columns = feature_names)
cust_all_cate_ohe_2018.info()
cust_all_cate_ohe_2018.head(10)


# In[27]:


#把連續型變數與類別型變數併成建模資料
model_data_2018 = pd.concat([cust_all_conts_s3_2018.reset_index(drop = True),
                             cust_all_cate_ohe_2018.reset_index(drop = True)],axis = 1)

print(model_data_2018.info())
print(model_data_2018.head(10))


# In[28]:


model_data2018_x_raw = model_data_2018.drop('BUY_INS2020',axis = 1)

x_2018 = model_data2018_x_raw

y_2018 = model_data_2018[['BUY_INS2020']]

x_train,x_test,y_train,y_test = train_test_split(x_2018,y_2018,test_size =0.3)

print('x_train_info')
print(x_train.head(10))
print('x_test_info')
print(x_test.head(10))
print('y_train_info')
print(y_train.head(10))
print('y_test_info')
print(y_test.head(10))

print(x_test.info())
print(y_test.info())


# In[29]:


#導入訓練完之2017年模型


# In[30]:


#讀取模型pickle
filename = '建模中間資料\\優化模型\\class_pred_model_store_ins.pkl'
model_storage = pkl.load(open(filename,'rb'))

#dt_model:決策樹模型
#dt_model_o:決策樹模型(oversampling)
#dt_model_u:決策樹模型(undersampling)
#rf:隨機森林模型
#rf_o:隨機森林模型(oversampling)
#rf_u:隨機森林模型(undersampling)
#logis_classifier:羅吉斯回歸模型
#logis_classifier_o:羅吉斯回歸模型(oversampling)
#logis_classifier_u:羅吉斯回歸模型(undersampling)
dt_model = model_storage[0]
dt_model_o = model_storage[1]
dt_model_u = model_storage[2]
rf = model_storage[3]
rf_o = model_storage[4]
rf_u = model_storage[5]
logis_classifier = model_storage[6]
logis_classifier_o = model_storage[7]
logis_classifier_u = model_storage[8]


# In[31]:


###對所有2018年流失客做預測


# In[32]:


#看precision recall accaracy
y_pred_dt_all = dt_model.predict(x_2018)
print('-------decision tree-------')
print(metrics.classification_report(y_pred_dt_all,y_2018))
print(confusion_matrix(y_pred_dt_all,y_2018))

y_pred_dto_all = dt_model_o.predict(x_2018)
print('-------decision tree oversampling-------')
print(metrics.classification_report(y_pred_dto_all,y_2018))
print(confusion_matrix(y_pred_dto_all,y_2018))

y_pred_dtu_all = dt_model_u.predict(x_2018)
print('-------decision tree undersampling-------')
print(metrics.classification_report(y_pred_dtu_all,y_2018))
print(confusion_matrix(y_pred_dtu_all,y_2018))

y_pred_rf_all = rf.predict(x_2018)
print('-------randon forest-------')
print(metrics.classification_report(y_pred_rf_all,y_2018))
print(confusion_matrix(y_pred_rf_all,y_2018))

y_pred_rfo_all = rf_o.predict(x_2018)
print('-------random forest oversampling-------')
print(metrics.classification_report(y_pred_rfo_all,y_2018))
print(confusion_matrix(y_pred_rfo_all,y_2018))


y_pred_rfu_all = rf_u.predict(x_2018)
print('-------random forest undersampling-------')
print(metrics.classification_report(y_pred_rfu_all,y_2018))
print(confusion_matrix(y_pred_rfu_all,y_2018))

y_pred_log_all = logis_classifier.predict(x_2018)
print('-------logistic regression-------')
print(metrics.classification_report(y_pred_log_all,y_2018))
print(confusion_matrix(y_pred_log_all,y_2018))

y_pred_logo_all = logis_classifier_o.predict(x_2018)
print('-------logistic regression oversampling-------')
print(metrics.classification_report(y_pred_logo_all,y_2018))
print(confusion_matrix(y_pred_logo_all,y_2018))

y_pred_logu_all = logis_classifier_u.predict(x_2018)
print('-------logistic regression undersampling-------')
print(metrics.classification_report(y_pred_logu_all,y_2018))
print(confusion_matrix(y_pred_logu_all,y_2018))


# In[33]:


#比較所有的模型 ROC 曲線
y_pred_proba_dt_2018_all = dt_model.predict_proba(x_2018)[::,1]
y_pred_proba_dt_o_2018_all = dt_model_o.predict_proba(x_2018)[::,1]
y_pred_proba_dt_u_2018_all = dt_model_u.predict_proba(x_2018)[::,1]
y_pred_proba_rf_2018_all = rf.predict_proba(x_2018)[::,1]
y_pred_proba_rf_o_2018_all = rf_o.predict_proba(x_2018)[::,1]
y_pred_proba_rf_u_2018_all = rf_u.predict_proba(x_2018)[::,1]
y_pred_proba_log_2018_all = logis_classifier.predict_proba(x_2018)[::,1]
y_pred_proba_log_o_2018_all = logis_classifier_o.predict_proba(x_2018)[::,1]
y_pred_proba_log_u_2018_all = logis_classifier_u.predict_proba(x_2018)[::,1]

dt_fpr,dt_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_dt_2018_all)
dto_fpr,dto_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_dt_o_2018_all)
dtu_fpr,dtu_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_dt_u_2018_all)

rf_fpr,rf_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_rf_2018_all)
rfo_fpr,rfo_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_rf_o_2018_all)
rfu_fpr,rfu_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_rf_u_2018_all)

log_fpr,log_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_log_2018_all)
logo_fpr,logo_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_log_o_2018_all)
logu_fpr,logu_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_log_u_2018_all)

auc_dt = metrics.roc_auc_score(y_2018,y_pred_proba_dt_2018_all)
auc_dt_o = metrics.roc_auc_score(y_2018,y_pred_proba_dt_o_2018_all)
auc_dt_u = metrics.roc_auc_score(y_2018,y_pred_proba_dt_u_2018_all)

auc_rf = metrics.roc_auc_score(y_2018,y_pred_proba_rf_2018_all)
auc_rf_o = metrics.roc_auc_score(y_2018,y_pred_proba_rf_o_2018_all)
auc_rf_u = metrics.roc_auc_score(y_2018,y_pred_proba_rf_u_2018_all)

auc_log = metrics.roc_auc_score(y_2018,y_pred_proba_log_2018_all)
auc_log_o = metrics.roc_auc_score(y_2018,y_pred_proba_log_o_2018_all)
auc_log_u = metrics.roc_auc_score(y_2018,y_pred_proba_log_u_2018_all)


plt.plot(dt_fpr,dt_tpr,marker = '.',label = "decision tree(AUC = %0.3f)" % auc_dt)
plt.plot(dto_fpr,dto_tpr,marker = '.',label = "decision tree(oversampling)(AUC = %0.3f)" % auc_dt_o)
plt.plot(dtu_fpr,dtu_tpr,marker = '.',label = "decision tree(undersampling)(AUC = %0.3f)" % auc_dt_u)

plt.plot(rf_fpr,rf_tpr,marker = '.',label = "random forest(AUC = %0.3f)" % auc_rf)
plt.plot(rfo_fpr,rfo_tpr,marker = '.',label = "random forest (oversampling)(AUC = %0.3f)" % auc_rf_o)
plt.plot(rfu_fpr,rfu_tpr,marker = '.',label = "random forest (undersampling)(AUC = %0.3f)" % auc_rf_u)


plt.plot(log_fpr,log_tpr,marker= '.',label = "logistic regression(AUC = %0.3f)" % auc_log)
plt.plot(logo_fpr,logo_tpr,marker= '.',label = "logistic regression (oversampling)(AUC = %0.3f)" % auc_log_o)
plt.plot(logu_fpr,logu_tpr,marker= '.',label = "logistic regression (undersampling)(AUC = %0.3f)" % auc_log_u)


plt.plot([0,1],[0,1],color = 'navy',lw=2,linestyle = '--')

plt.title('ROC Plot')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()


# In[34]:


#各模型最佳的ROC曲線(AUC最大的)
y_pred_proba_dtu_2018_all = dt_model_u.predict_proba(x_2018)[::,1]
y_pred_proba_rfu_2018_all = rf_u.predict_proba(x_2018)[::,1]
y_pred_proba_log_2018_all = logis_classifier.predict_proba(x_2018)[::,1]


dtu_fpr,dtu_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_dtu_2018_all)
rfu_fpr,rfu_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_rfu_2018_all)
log_fpr,log_tpr, _ = metrics.roc_curve(y_2018,y_pred_proba_log_2018_all)

auc_dtu_2018_all = metrics.roc_auc_score(y_2018,y_pred_proba_dtu_2018_all)
auc_rfu_2018_all = metrics.roc_auc_score(y_2018,y_pred_proba_rfu_2018_all)
auc_log_2018_all = metrics.roc_auc_score(y_2018,y_pred_proba_log_2018_all)

plt.plot(dtu_fpr,dtu_tpr,marker = '.',label = "decision tree (undersampling)(AUC = %0.3f)" % auc_dtu_2018_all)
plt.plot(rfu_fpr,rfu_tpr,marker = '.',label = "random forest (undersampling)(AUC = %0.3f)" % auc_rfu_2018_all)
plt.plot(log_fpr,log_tpr,marker= '.',label = "logistic regression (AUC = %0.3f)" % auc_log_2018_all)
plt.plot([0,1],[0,1],color = 'navy',lw=2,linestyle = '--')

plt.title('ROC Plot 2018')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()


# In[32]:


#####5.模型應用(驗證實際再購率)


# In[33]:


#用AUC最大的為最終應用的模型(用羅吉斯回歸為後續應用模型)


# In[34]:


##以2018年全部流失客資料來看


# In[35]:


print(model_data_2018.head(5))
model_data_2018.BUY_INS2020.value_counts()


# In[39]:


##計算客戶再購保險比例##(隨機狀況下)
model_data_app_x_2018 = model_data_2018.drop('BUY_INS2020',axis = 1)

x_app2018 = model_data_app_x_2018

y_app2018 = model_data_2018[['BUY_INS2020']]

#產生預測流失客再購機率
log_pred_probs2018 = logis_classifier.predict_proba(x_app2018)

model_app2018 = pd.DataFrame(log_pred_probs2018,columns = ['predict_not_buy','predict_buy'])
#產生預測流失客是否再購
y_app_pred2018 = logis_classifier.predict(x_app2018)

model_predict_cate2018 = pd.DataFrame(y_app_pred2018,columns = ['model_predict_cate'])

model_app2_2018 = pd.concat([model_app2018.reset_index(drop = True),
                             model_predict_cate2018.reset_index(drop = True)],axis = 1)

model_app3_2018 = pd.concat([model_app2_2018.reset_index(drop=True),
                        y_app2018.reset_index(drop = True)],axis = 1)

model_app3_2018['client_count'] = 1

#客戶人數累積加總
model_app3_2018['client_cum_sum'] = model_app3_2018['client_count'].cumsum()
#客戶再購保險累積加總
model_app3_2018['client_buy_cum'] = model_app3_2018['BUY_INS2020'].cumsum()
#客戶累積再購保險比例
model_app3_2018['client_buy_c_rate'] = model_app3_2018['client_buy_cum'] / model_app3_2018['client_cum_sum'] 
print(model_app3_2018.head(100))

plt.plot(model_app3_2018['client_cum_sum'],
         model_app3_2018['client_buy_c_rate'])


# In[40]:


##依據模型產出之保險再購機率做排序##

model_app4_2018 = model_app3_2018.sort_values(by = ['predict_buy'],ascending = True)

model_app4_2018['client_cum_sum_p'] = model_app4_2018['client_count'].cumsum()
model_app4_2018['client_buy_cum_p'] = model_app4_2018['BUY_INS2020'].cumsum()
model_app4_2018['client_buy_c_rate_p'] = model_app4_2018['client_buy_cum_p'] / model_app4_2018['client_cum_sum_p'] 
print(model_app4_2018.head(100))


# In[41]:


##藉由資料切割來讓曲線呈現上漲趨勢(主要實際再購率趨勢圖形，並匯出可畫excel圖形的 2018年再購保險模型應用0220.csv檔案)


# In[43]:


##看各群組的各別再購率

#1.model result cut cluster(用分位數做組的切割)
SEG_label = ['SEG1','SEG2','SEG3','SEG4','SEG5','SEG6','SEG7','SEG8','SEG9','SEG10']
model_app4_2018['seg_label'] = pd.qcut(model_app4_2018['client_cum_sum_p'],
                                 q = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                                 labels = SEG_label)

group_m_count = pd.DataFrame(model_app4_2018['seg_label'].value_counts())

#各群各別計算實際再購率
group_m = model_app4_2018[['BUY_INS2020','seg_label']].loc[model_app4_2018.BUY_INS2020 == 1].groupby('seg_label').agg(['count'])
group_m.reset_index()

#各群再購率計算(模型)
group_m_df = pd.concat([group_m_count,group_m],axis = 1)
group_m_df.columns = ['model_seg_count','model_seg_buy']
group_m_df['CR_M'] = group_m_df['model_seg_buy'] / group_m_df['model_seg_count']
print('---模型各群組實際再購率---')
print(group_m_df)
print('\n')


#2.random result cut cluster
SEG_label_R = ['SEG1','SEG2','SEG3','SEG4','SEG5','SEG6','SEG7','SEG8','SEG9','SEG10']
model_app4_2018['seg_label_R'] = pd.qcut(model_app4_2018['client_cum_sum'],
                                 q = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                                 labels = SEG_label_R)

group_r_count = pd.DataFrame(model_app4_2018['seg_label_R'].value_counts())
#print(group_r_count)

#各群各別計算再購
group_r = model_app4_2018[['BUY_INS2020','seg_label_R']].loc[model_app4_2018.BUY_INS2020 == 1].groupby('seg_label_R').agg(['count'])
group_r.reset_index()

#各群在購率計算(隨機)
group_r_df = pd.concat([group_r_count,group_r],axis = 1)
group_r_df.columns = ['random_seg_count','random_seg_buy']
group_r_df['CR_R'] = group_r_df['random_seg_buy'] / group_r_df['random_seg_count']
print('---各群組實際再購率(隨機選擇)---')
print(group_r_df)
print('\n')

#合併model分群與隨機分群的結果
final_rebuy_rate = pd.merge(group_m_df,group_r_df,left_index = True,right_index = True)
final_rebuy_rate['segement'] = final_rebuy_rate.index
final_rebuy_rate['seg_num'] = final_rebuy_rate['segement'].str.partition("SEG",True)[2]
#print(final_rebuy_rate.head(10))

final_rebuy_rate['seg_num'] = final_rebuy_rate['seg_num'].astype('int')
#print(final_rebuy_rate.info())
final_rebuy_rate2 = final_rebuy_rate.sort_values(by = 'seg_num')

print('各群組實際再購率(模型與隨機比較)')
print(final_rebuy_rate2)
print('\n')

final_rebuy_rate2['比例'] = ['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']

plt.plot(final_rebuy_rate2['比例'],
         final_rebuy_rate2['CR_M'],
         color = 'red',label = 'model prediction')

plt.plot(final_rebuy_rate2['比例'],
         final_rebuy_rate2['CR_R'],
        color = 'blue',label = 'random prediction')


plt.grid(axis = 'x',color = '0.95')
plt.legend(title = 'accumulate buy rate')

final_rebuy_rate2.to_csv('建模中間資料\\優化模型\\2018年再購保險模型應用0220.csv',encoding = 'utf_8_sig')


# In[45]:


#####5.產出後續模型標籤的合併大表


# In[46]:


#NEW WAY


# In[47]:


#####編制模型貼標後最後表格#####
#1.存成CSV檔，要匯出到MYSQL進行後續資料應用

#需包含欄位與資料：
#1.要同時包含流失客、活躍客

#順序
#1.實際再購各產品  
#2.分群 
#3.模型流失客未再購機率、模型流失客再購機率、模型流失客再購與否預測、SEG_LABEL  
#4.其他多增加的資料欄位(2020年年齡、再購保險的重新標記)


# In[54]:


#最佳的分類預測模型是羅吉斯回歸，用此做預測

#需要欄位：流失客再購及未再購預測機率及預測類別
y_pred_proba_log_2018 = logis_classifier.predict_proba(x_2018)

y_pred_log_2018 = logis_classifier.predict(x_2018)

y_pred_proba_log_df_2018 = pd.DataFrame(y_pred_proba_log_2018,columns = ['流失客未再購機率','流失客再購機率'])

y_pred_log_df_2018 = pd.DataFrame(y_pred_log_2018,columns = ['流失客再購與否預測'])


#串接流失客的衍生欄位
#1.串預測機率
raw_pred_2018 = pd.concat([cust_all_lose_2018.drop(['BUY_INS2020','year_old_2020'],axis = 'columns').reset_index(drop=True),
                            y_pred_proba_log_df_2018.reset_index(drop = True)],axis = 1)

#2.串預測類別
raw_pred2_2018 = pd.concat([raw_pred_2018.reset_index(drop = True),
                       y_pred_log_df_2018.reset_index(drop = True)],
                        axis = 1)

raw_pred2_2018_sort = raw_pred2_2018.sort_values(by = ['流失客再購機率'],ascending = True)

SEG_label = ['SEG1','SEG2','SEG3','SEG4','SEG5','SEG6','SEG7','SEG8','SEG9','SEG10']
raw_pred2_2018_sort['seg_label'] = pd.qcut(raw_pred2_2018_sort['流失客再購機率'],
                                 q = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                                 labels = SEG_label)


print(raw_pred2_2018_sort.tail(100))


# In[52]:


raw_pred3_2018 = raw_pred2_2018_sort[['CUST_COD_RAW','流失客未再購機率','流失客再購機率','流失客再購與否預測','seg_label']]

#testing:確認原始資料與預測資料順序一致
#print(cust_all_lose.tail(10))

#print(raw_pred2.tail(10))

#串活躍客+流失客所有客戶資料
#3.
merge_all_model_tags = pd.merge(cust_all_alive_lose_2018,
                                raw_pred3_2018,
                                left_on = 'CUST_COD_RAW',
                                right_on = 'CUST_COD_RAW',
                                how = 'left')

merge_all_model_tags['year_old_2020'] = round((pd.to_datetime('2020-12-31') - pd.to_datetime(merge_all_model_tags['BIRTH_DATE_1'],format = '%Y%m%d'))/np.timedelta64(1,'Y'),0)

merge_all_model_tags['ID2020'] = merge_all_model_tags['ID2020'].astype(str)

merge_all_model_tags['ID_INS2020'] = merge_all_model_tags['ID_INS2020'].astype(str)

merge_all_model_tags['BUY2020_INS'] = np.where(merge_all_model_tags['ID_INS2020'] == "nan",0,1)

print(merge_all_model_tags.head(1))


# In[50]:


merge_all_model_tags.to_csv('建模中間資料\\優化模型\\2018年流失客再購保險分群預測總表0220.csv',encoding = 'utf_8_sig')

