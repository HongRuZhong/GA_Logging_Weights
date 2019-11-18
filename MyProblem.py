import numpy as np
import geatpy as ea
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
import My_PythonLib as MP
import time
import os
def Load_Data(ipath,remainlabel=None):
	train=np.loadtxt(ipath+"train.txt",skiprows=1,dtype=str)
	test=np.loadtxt(ipath+"test.txt",skiprows=1,dtype=str)
	data=np.row_stack((train[:,3:].astype(float),test[:,3:].astype(float)))
	data[:,:-1]=StandardScaler().fit_transform(data[:,:-1])
	if remainlabel is not None:  #如果是单一类就设置为1，否则为0
		for x in data:
			if x[-1]==remainlabel:
				x[-1]=1
			else:
				x[-1]=0
	train=data[0:3122]  #划分训练集
	# print(train[0:50])
	# print("no 0",np.count_nonzero(train[:,-1]))
	test=data[3122:]
	# print(train.dtype,train[-1],test[0],111)
	# train,test=train_test_split(data,test_size=0.3,random_state=1)
	return train,test
def Load_8wData(iapth):
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

def test():
	ipath="D:\\Data\\Data_机器学习常用数据集\\3口井共3000多样本岩性数据\\"

	# ipath="D:\\Data\\Data_机器学习常用数据集\\8W89口井岩性数据\\Data_去除678类.txt"
	train_data,test_data=Load_Data(ipath)
	# print(train_data[0:5],test_data[0:5])
	# train_data,test_data=Load_8wData(ipath)
	mean,var=Get_Mean_Std(train_data)
	# print(mean)
	# print(mean,var)
	GM=MP.Gaussian_Membership()
	# GM=MP.Knn_With_Weights(5)
	GM.fit(train_data[:,0:-1],train_data[:,-1])
	# weight=np.loadtxt("D:\\sudty\\遗传算法_岩性权重\\3口井权重结果\\精度变化曲线\\best_weight.txt")
	# weight=weight[:,0:6]
	weight=np.full((6,6),1)
	# print(weight)

	pred=GM.predict(test_data[:,0:-1],weight)
	print(accuracy_score(test_data[:,-1],pred))

def Get_Confm():
	'''输出混淆矩阵'''
	# weight_path = "D:\\sudty\\遗传算法_岩性权重\\3口井权重结果\\高斯隶属度result\\"
	weight_path="D:\\sudty\\遗传算法_岩性权重\\3口井权重结果\\KNN精度变化曲线\\"
	labeldata=np.loadtxt(weight_path+"KNN最好预测标签.txt",skiprows=1)
	ortlabel=labeldata[:,0]
	bestacclabel=labeldata[:,-1]
	cm=confusion_matrix(ortlabel,bestacclabel)
	cm=MP.MakeConfusionMatrixWithACC(cm)
	cm.to_csv(weight_path+"混淆矩阵.csv")
	# np.savetxt(weight_path+"混淆矩阵.txt",cm,fmt='%.04f',delimiter='\t')


def Get_Gaussian_Model(remainlabel=None):
	ipath="D:\\Data\\Data_机器学习常用数据集\\3口井共3000多样本岩性数据\\"
	opath = "D:\\sudty\\遗传算法_岩性权重\\3口井权重结果\\"
	# ipath = "D:\\Data\\Data_机器学习常用数据集\\8W89口井岩性数据\\Data_去除678类.txt"
	train_data, test_data = Load_Data(ipath, remainlabel)
	# print(test_data)
	# train_data,test_data=Load_8wData(ipath)
	mean, var = Get_Mean_Std(train_data)
	# print(mean)
	# print(mean,var)
	GM = MP.Gaussian_Membership()
	GM.fit(mean, var)
	#输出等权重的混淆矩阵
	pred=GM.predict(test_data[:,0:-1])
	cm=confusion_matrix(test_data[:,-1],pred)
	cm=MP.MakeConfusionMatrixWithACC(cm)
	cm.to_csv(opath+"等权fuzzy混淆矩阵.csv")
	# # train_data,test_data=Load_Data(ipath)
	# train_data, test_data = Load_8wData(ipath)
	# mean, var = Get_Mean_Std(train_data)
	# # print(mean,var)
	# GM = MP.Gaussian_Membership()
	# GM.fit(mean, var)
	return GM,test_data

def Get_WeightKNN_Model(remainlabel=None):
	ipath="D:\\Data\\Data_机器学习常用数据集\\3口井共3000多样本岩性数据\\"
	opath="D:\\sudty\\遗传算法_岩性权重\\3口井权重结果\\"
	# ipath = "D:\\Data\\Data_机器学习常用数据集\\8W89口井岩性数据\\Data_去除678类.txt"
	train_data, test_data = Load_Data(ipath, remainlabel)
	GM = MP.Knn_With_Weights(5)
	GM.fit(train_data[:,0:-1],train_data[:,-1] )
	#输出等权重的混淆矩阵
	pred=GM.predict(test_data[:,0:-1])
	cm=confusion_matrix(test_data[:,-1],pred)
	cm=MP.MakeConfusionMatrixWithACC(cm)
	cm.to_csv(opath+"等权KNN混淆矩阵.csv")
	# # train_data,test_data=Load_Data(ipath)
	# train_data, test_data = Load_8wData(ipath)
	# mean, var = Get_Mean_Std(train_data)
	# # print(mean,var)
	# GM = MP.Gaussian_Membership()
	# GM.fit(mean, var)
	return GM,test_data

def Predict_SingleLabel_Weights():
	'''计算单独类训练，每次迭代权重的预测精度'''
	ipath="D:\\Data\\Data_机器学习常用数据集\\3口井共3000多样本岩性数据\\"
	# ipath = "D:\\Data\\Data_机器学习常用数据集\\8W89口井岩性数据\\Data_去除678类.txt"
	weight_path="D:\\sudty\\遗传算法_岩性权重\\3口井权重结果\\KNN精度变化曲线\\单独类预测结果\\"
	weights=np.loadtxt(weight_path+"DE_bin_单独类别训练最好精度所有类迭代记录.txt")
	train_data, test_data = Load_Data(ipath)
	mean, var = Get_Mean_Std(train_data)
	GM = MP.Gaussian_Membership()
	GM=MP.Knn_With_Weights(5)
	GM.fit(train_data[:,0:-1],train_data[:,-1])

	pred_list=[]
	for x in weights:
		x=np.reshape(x,[6,6])
		pred = GM.predict(test_data[:, :-1], x)  # 选择模型
		acc = accuracy_score(pred, test_data[:, -1])
		pred_list.append(acc)
	No_array = np.arange(len(weights)).astype(np.float)
	Weights_Iter=np.column_stack((No_array,np.array(pred_list)))
	np.savetxt(os.path.join(weight_path, "DE_bin_单独类别训练_所有迭代权重预测精度.txt"), Weights_Iter, fmt="%.04f",  delimiter="\t")

BestACC_list=[] #记录遗传算法过程的最大目标函数值
AvgACC_list=[]#记录遗传算法过程的平均目标函数值
BestWeight_each_iter=[] #记录每次优化迭代过程中最好的权重
BestWeight=0
class MyProblem_SingleAim(ea.Problem):
	'''带约束的单目标优化_'''
	def __init__(self,remainlabel=None):

		name='MyProblem' #初始化name（函数名称，可以随意设置）
		M=1 #初始化M(目标维数，目标函数值只有一个）
		maxormins=[-1] #初始化目标最小最大化标记列表，1：min;-1:max
		self.remainlabel=remainlabel
		if self.remainlabel is None:
			self.Dim=36#初始化Dim（决策变量维数)
		else:
			self.Dim=12
		varTypes=[0]*self.Dim#初始化决策变量类型，0：连续；1：离散
		lb=np.full((self.Dim,),0) #决策变量下界
		ub=np.full((self.Dim,),1) #决策变量上届
		lbin=np.full((self.Dim,),1) #决策变量上届是否包含边界，1：包含；0：不包含
		ubin=np.full((self.Dim,),1)  #决策变量下届是否包含边界，1：包含；0：不包含
		self.bestweighteachiter=0
		self.model,self.test_data=Get_Gaussian_Model(remainlabel)
		self.KNN,self.test_data=Get_WeightKNN_Model(remainlabel)
		self.youhuatime=time.time()
		#调用父类构造函数完成实例化
		ea.Problem.__init__(self,name,M,maxormins,self.Dim,varTypes,lb,ub,lbin,ubin)

		self.best_acc=0
		self.best_acc_pred=0
		self.best_weights=0

	def aimFunc(self,pop): #目标函数，pop为传入的种群对象
		start_time=time.time()
		# print("优化时间：",start_time-self.youhuatime)
		Vars=pop.Phen #得到决策变量矩阵
		# print("Vars:",Vars.shape)
		weight_list=[]
		for x in Vars:
			weight_one=np.reshape(x,(int(self.Dim/6),6))
			weight_list.append(weight_one)
		acc_list=[]
		for x in weight_list:
			lt=time.time()
			pred=self.KNN.predict(self.test_data[:,:-1],x)  #选择模型
			acc=accuracy_score(pred,self.test_data[:,-1])
			if acc>self.best_acc:
				self.best_acc=acc
				self.best_acc_pred=pred
				self.best_weights=np.ravel(x)

			et=time.time()-lt
			# print(recall_score(pred,self.test_data[:,-1],average=None))
			acc_list.append(acc)  #召回率返回的是一个list，取最后一个是我们要的召回率

		#计算目标函数值，赋值给pop种群对象的objv属性
		pop.ObjV=np.row_stack(acc_list)
		end_time=time.time()
		self.youhuatime=time.time()
		AvgACC_list.append(np.average(pop.ObjV))
		BestACC_list.append(np.max(pop.ObjV))
		print("spenttime:{},最好精度：{}".format(end_time-start_time,np.max(pop.ObjV)))
		self.beatweight=Vars[np.argmax(pop.ObjV)][6:]

		#判断是否是单一类别
		if self.remainlabel is not None:
			self.bestweighteachiter = Vars[np.argmax(pop.ObjV)][6:]
			BestWeight_each_iter.append(self.bestweighteachiter)
		#采用可行性法则处理约束，生成种群个体违反约束程度矩阵
		# pop.CV=np.column_stack([np.abs(np.sum(Vars[:,0:6],axis=1)-1),
		# 				  np.abs(np.sum(Vars[:, 6:12], axis=1) - 1),
		# 				  np.abs(np.sum(Vars[:, 12:18], axis=1) - 1),
		# 				  np.abs(np.sum(Vars[:, 18:24], axis=1) - 1),
		# 				  np.abs(np.sum(Vars[:, 24:30], axis=1) - 1),
		# 				  np.abs(np.sum(Vars[:, 30:36], axis=1) - 1),
		# 				  ])#第三个约束
		# # print(np.sum(Vars[:,0:6],axis=1).shape)
		# print("Vars[:,0:6]:",type(Vars[:,0:6]),Vars[:,0:6].shape,Vars[:,0:6])
		# print("pop.CV:", type(pop.CV), pop.CV.shape, pop.CV)

if __name__=="__main__":
	# test()
	# Predict_SingleLabel_Weights()
	# Get_Confm()
	# Get_WeightKNN_Model()
	Get_Gaussian_Model()