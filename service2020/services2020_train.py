#coding:utf-8
import pandas as pd
#from pandas import Series,DataFrame
import numpy as np
import tensorflow as tf
import pickle
from dense import densenet
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler
from datapreprocess import data_preprocess_years,data_preprocess,table_merge


#读取原始数据

val_data_base=pd.read_csv('./data2020/base_verify1.csv',encoding='gbk')
val_data_money=pd.read_csv('./data2020/money_information_verify1.csv',encoding='gbk')
val_data_patent=pd.read_csv('./data2020/paient_information_verify1.csv',encoding='gbk')
val_data_report=pd.read_csv('./data2020/year_report_verify1.csv',encoding='gbk')
#四张表合并成两张表
val_data=table_merge(val_data_base,val_data_patent,['ID'])
val_data_years=table_merge(val_data_money,val_data_report,['ID','year'])
#基础验证集处理
df_val_data=val_data.dropna(subset=['flag'])#去除某列中有空值项
df_val_data=data_preprocess(df_val_data)
df_val_data=df_val_data.drop(['控制人ID'],axis=1)#去除控制人id
#年度验证集处理
df_val_data_years=data_preprocess_years(val_data_years)

list_data=[]#存放数据
list_label=[]#存放flag
#基础验证表和年份表合并
train_data=pd.merge(df_val_data,df_val_data_years,on='ID')
train_data=train_data.set_index('ID',drop=True)#ID项移作索引值
#标签单独分离出来
df_flag=train_data.flag#取出标签值
train_data=train_data.drop(['flag'],axis=1)#去掉标签值项
train_data.insert(loc=25,column='flag',value=df_flag)#加上标签值项
#数据按行去读取

for indexs in train_data.index:
    data=train_data.loc[indexs][0:-1]#data是series
    np_data=np.array(data)
  #  np_data2=np_data.reshape([5,5,1])
    np_data = np.reshape(np_data,(5, 5, 1))
    #np_data1=np.pad(np_data,(0,7),mode='constant')一位数组填充
    # p=[0,0,0,0,0,0,0]
    # p_arr = np.concatenate((np_data, p))
    list_data.append(np_data)
    list_label.append(int(train_data.loc[indexs]['flag']))#打上标签

list_label_onehot=tf.one_hot(list_label,depth=2)#标签转换为onehot编码


db_train=tf.data.Dataset.from_tensor_slices((list_data[:15000],list_label_onehot[:15000]))
db_val=tf.data.Dataset.from_tensor_slices((list_data[15000:25000],list_label_onehot[15000:25000]))
db_test=tf.data.Dataset.from_tensor_slices((list_data[25000:],list_label_onehot[25000:]))
db_train=db_train.shuffle(30000).batch(128)
db_val=db_val.batch(128)
db_test=db_test.batch(8000)
model=densenet()
recoder=model.fit(db_train,validation_data=db_val,validation_freq=1,epochs=100).history
model.save(f'./data2020/dense5x5_100epochs.h5')
test_acc=model.evaluate(db_test)
print('---model have saved---')
print(recoder['val_accuracy'])
print(recoder['accuracy'])
plt.figure()
returns=recoder['val_accuracy']
plt.plot(np.arange(len(returns)),returns,label='Val_accuracy')
returns=recoder['accuracy']
plt.plot(np.arange(len(returns)),returns,label='Train_accuracy')
plt.plot([len(returns)-1],[test_acc[-1]],'D',label='Test_accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('./data2020/train.svg')
plt.show()




#（N/A）行业	区域	企业类型	控制人类型	控制人ID	控制人持股比例	flag  专利	商标	著作权

#year	从业人数	资产总额	负债总额	营业总收入	主营业务收入	利润总额	净利润	纳税总额	所有者权益合计
#year	债权融资额度	债权融资成本	股权融资额度	股权融资成本	内部融资和贸易融资额度	内部融资和贸易融资成本	项目融资和政策融资额度	项目融资和政策融资成本



