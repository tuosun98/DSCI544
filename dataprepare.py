
import pandas as pd
import numpy as np
import os
analystdata=pd.read_csv('data/analyst_detail.csv',sep='|', encoding='GB18030')
predictday=60
trainend=20200631
predthing='return'    #return,label,expreturn
train_data=analystdata.loc[analystdata['period']<=trainend]
labelname='ret'+str(predictday)+'end_extrareturn'
train_data.rename(columns={labelname:'ret'},inplace=True )
train_data.rename(columns={'allreport':'words'},inplace=True )
train_data=train_data[['ret','words']]
train_data=train_data.dropna(0)
if (predthing=='label'):
	train_data['label']=0
	train_data['label']=(train_data['ret']>0).apply(lambda x: int(x))
else:
	train_data['label']=train_data['ret']
train_data=train_data.sample(frac=1)
train_dataout=train_data.iloc[0:int(0.8*len(train_data))]
val_dataout=train_data.iloc[int(0.8*len(train_data)):len(train_data)]
os.mkdir('data3')
train_dataout.to_csv('data3/train.tsv', sep='\t', index=False, columns=["label","words"], mode="w")

val_dataout.to_csv('data3/dev.tsv', sep='\t', index=False, columns=["label","words"], mode="w")

train_data=analystdata.loc[analystdata['period']>trainend]
labelname='ret'+str(predictday)+'end_extrareturn'
train_data.rename(columns={labelname:'ret'},inplace=True )
train_data.rename(columns={'allreport':'words'},inplace=True )
#train_data=train_data[['ret','words']]
train_data=train_data.dropna(0)
if (predthing=='label'):
	train_data['label']=0
	train_data['label']=(train_data['ret']>0).apply(lambda x: int(x))
else:
	train_data['label']=train_data['ret']
train_data.to_csv('data3/test.tsv', sep='\t', index=False, columns=["label","words"],  mode="w")