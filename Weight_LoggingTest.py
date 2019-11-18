'''求出权重后，用8w数据进行测试'''
import numpy as np
import My_PythonLib as MP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
# def Load_Data(ipath):
# 	train=np.loadtxt(ipath+"train.txt",skiprows=1,dtype=str)
# 	test=np.loadtxt(ipath+"test.txt",skiprows=1,dtype=str)
# 	data=np.row_stack((train[:,3:].astype(float),test[:,3:].astype(float)))
# 	data[:,:-1]=StandardScaler().fit_transform(data[:,:-1])
# 	train,test=train_test_split(data,test_size=0.3,random_state=1)
# 	return train,test
def Load_Data(iapth):
	data=np.loadtxt(ipath,skiprows=1,dtype=str)
	data=data[:,3:].astype(float)
	data[:,:-1]=StandardScaler().fit_transform(data[:,:-1])
	train,test=train_test_split(data,test_size=0.3,random_state=1)
	return train,test
def Get_Mean_Std(data):
	label=data[:,-1]
	lable_No=np.unique(label)
	mean_list=[]
	var_list=[]
	for x in lable_No:
		ll=[y for y in data if y[-1]==x]
		ll=np.row_stack(ll)
		mean=np.mean(ll[:,0:-1],axis=0)
		mean_list.append(mean)
		var=np.var(ll[:,0:-1],axis=0)
		var_list.append(var)
	return np.row_stack(mean_list),np.row_stack(var_list)

def Load_Weight(ipath):
	ww=np.loadtxt(ipath+"权重.txt",dtype=float)
	ww=np.reshape(ww,(6,6))
	return ww

# ipath="D:\\Data\\Data_机器学习常用数据集\\3口井共3000多样本岩性数据\\"
ipath="D:\\Data\\Data_机器学习常用数据集\\8W89口井岩性数据\\Data_去除678类.txt"
wpath="D:\\sudty\\遗传算法_岩性权重\\"
weight=Load_Weight(wpath)
train_data,test_data=Load_Data(ipath)
mean,var=Get_Mean_Std(train_data)
# print(mean,var)
GM=MP.Gaussian_Membership()
GM.fit(mean,var)
pred=GM.predict(test_data[:,0:-1])
st=time.time()
pred_w=GM.predict(test_data[:,0:-1],weight)
print("time:",time.time()-st)
# MP.ML_Model_Run_NoTree(train_data[:,0:-1],train_data[:,-1],test_data[:,0:-1],test_data[:,-1])
print(accuracy_score(test_data[:,-1],pred))
print(accuracy_score(test_data[:,-1],pred_w))
