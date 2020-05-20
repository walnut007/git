#coding:utf-8
import pandas as pd
#from pandas import Series,DataFrame
import numpy as np
import tensorflow as tf
import pickle
from dense import densenet
#from sklearn.preprocessing import MinMaxScaler
from datapreprocess import data_preprocess_years,data_preprocess,table_merge
from tensorflow import keras

#读取原始数据
# test_data_base=pd.read_csv('./data2020/base_train_sum.csv',encoding='gbk')
# test_data_patent=pd.read_csv('./data2020/knowledge_train_sum.csv',encoding='gbk')
# test_data_money=pd.read_csv('./data2020/money_report_train_sum.csv',encoding='gbk')
# test_data_report=pd.read_csv('./data2020/year_report_train_sum.csv',encoding='gbk')
test_data_base=pd.read_csv('./data2020/base_verify1.csv',encoding='gbk')
test_data_money=pd.read_csv('./data2020/money_information_verify1.csv',encoding='gbk')
test_data_patent=pd.read_csv('./data2020/paient_information_verify1.csv',encoding='gbk')
test_data_report=pd.read_csv('./data2020/year_report_verify1.csv',encoding='gbk')
#四张表合并成两张表
test_data=table_merge(test_data_base,test_data_patent,['ID'])
test_data_years=table_merge(test_data_money,test_data_report,['ID','year'])
#基础验证集处理
df_test_data=data_preprocess(test_data)
df_test_data=df_test_data.drop(['控制人ID'],axis=1)#去除控制人id
#年度验证集处理
df_test_data_years=data_preprocess_years(test_data_years)

list_predict_label=[]#存放flag
list_id=[]
list_predict_prob=[]
#基础验证表和年份表合并
test_data=pd.merge(df_test_data,df_test_data_years,on='ID')
test_data=test_data.set_index('ID',drop=True)#ID项移作索引值
#标签单独分离出来
#数据按行去读取
result = pd.DataFrame(columns = ['ID','Predict_prob','Predict_flag'])
#读取训练好的模型
new_model=keras.models.load_model(f'./loadmodels/dense5x5_100epochs.h5')
for indexs in test_data.index:
    list_id.append(indexs)
    data=test_data.loc[indexs][0:-1]#data是series
    np_data=np.array(data)
    np_data = np.reshape(np_data,(5, 5, 1))
    list_data=[]
    list_data.append(np_data)
    db_test=tf.data.Dataset.from_tensor_slices(list_data).batch(2)
    predict_label=new_model.predict(db_test)
    if(predict_label[0][0]>predict_label[0][1]):
        list_predict_prob.append(predict_label[0][0])
        list_predict_label.append(0)  # 打上标签
        predict_prob=predict_label[0][0]
       # predict_label = 0
    else:
        list_predict_prob.append(predict_label[0][1])
        list_predict_label.append(1)  # 打上标签
        predict_prob = predict_label[0][1]
       # predict_label = 1
  #  predict_label=np.argmax(predict_label,axis=1)
   # list_predict_label.append(predict_label)#打上标签
result=result.append(pd.DataFrame({'ID':list_id,'Predict_prob':list_predict_prob,'Predict_flag':list_predict_label}),ignore_index=False)


result.to_csv('./data2020/result/predict_label_test.csv')
print('---pridect labels have finished---')









