import numpy as np
import pandas as pd
def data_preprocess(data):
    df_data=pd.DataFrame(data)#转换为DataFrame
    #df_val_data=df_val_data.dropna()#去除有空值项 删除所有空值的记录
    df_data=df_data.fillna({'注册时间':2008,'注册资本':np.mean(df_data['注册资本'])})#填充空值项，注册资本用平均值去填充 年份最早2000年，最晚2014
    df_data=df_data.fillna(0)
    #行业数据转换为数字 df_val_data2['区域'].unique()
    df_data.loc[df_data['行业']=='商业服务业','行业']=1
    df_data.loc[df_data['行业']=='交通运输业','行业']=2
    df_data.loc[df_data['行业']=='工业','行业']=3
    df_data.loc[df_data['行业']=='零售业','行业']=4
    df_data.loc[df_data['行业']=='社区服务','行业']=5
    df_data.loc[df_data['行业']=='服务业','行业']=6
    #企业类型转换
    df_data.loc[df_data['企业类型']=='股份有限公司','企业类型']=1
    df_data.loc[df_data['企业类型']=='农民专业合作社','企业类型']=2
    df_data.loc[df_data['企业类型']=='集体所有制企业','企业类型']=3
    df_data.loc[df_data['企业类型']=='合伙企业','企业类型']=4
    df_data.loc[df_data['企业类型']=='有限责任公司','企业类型']=5
    #控制人类型转换
    df_data.loc[df_data['控制人类型']=='自然人','控制人类型']=1
    df_data.loc[df_data['控制人类型']=='企业法人','控制人类型']=2
    #去除一些无关项目
    df_data=df_data.drop(['注册时间','区域'],axis=1)#,'控制人ID'
    return df_data

def data_preprocess_years(data):
    data = data.fillna(0)
    data = data.drop(['year'], axis=1)
    data = data.groupby(by='ID').sum()  # 若干年的数据求和合并成一行
    return data

#合并两张表，list为合并共同项
def table_merge(table_1,table_2,list):
    table=pd.merge(table_1,table_2,on=list)
    return table

# def normal_data(data,m):
#     scaler=MinMaxScaler()
#     scaler.fit(data)
#     scaler.data_max_
#     normal_data=scaler.transform(data)
#     normal_data=np.pad(normal_data,pad_width=((0,0),(m-len(data))),mode='constant')
#     normal_data=np.reshape(normal_data,(m,m,1))
#     return normal_data