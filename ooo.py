import numpy as np
import geatpy as ea
import GA_Logging_Weights.MyProblem as MyProblem
from GA_Logging_Weights.Steady_GA import Mysoea_steadyGA_templet
import GA_Logging_Weights.GA_Model as Models
import os
import time

def GA_Run():
	problem=MyProblem.MyProblem_SingleAim() #实例化问题对象

	"""==============================种群设置==========================="""
	Encodings='RI'  #编码方式，很坑爹，必须用单引号
	NIND=50 #种群大小
	Field=ea.crtfld(Encodings,problem.varTypes,problem.ranges,problem.borders) #创建区域描述器
	population=ea.Population(Encodings,Field,NIND) #实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
	"""===========================算法参数设置=========================="""
	# myAlgorithm=Mysoea_steadyGA_templet(problem,population)#实例化一个算法模板对象
	# myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # 实例化一个算法模板对象
	myAlgorithm = ea.soea_SGA_templet(problem, population)  # 实例化一个算法模板对象
	myAlgorithm.MAXGEN=1000 #最大遗传代数
	# myAlgorithm.mutOper.F = 0.5 # 设置差分进化的变异缩放因子
	# myAlgorithm.recOper.XOVR=0.5 #设置交叉概率
	myAlgorithm.drawing=1 #设置绘图方式
	"""=====================调用算法模板进行种群进化====================="""
	[population,obj_trace,var_trace]=myAlgorithm.run() #执行算法模板
	#输出结果
	best_gen=np.argmax(obj_trace[:,1]) #记录最优种群是在那一代
	best_Objv=obj_trace[best_gen,1]
	print('最优的目标函数值为：%s'%(best_Objv))
	print('最优的决策变量值为：')
	for i in range(var_trace.shape[1]):
		print(var_trace[best_gen, i])
	print('有效进化代数：%s'%(obj_trace.shape[0]))
	print('最优的一代是第%s 代'%(best_gen + 1))
	print('评价次数：%s'%(myAlgorithm.evalsNum))
	print('时间已过%s 秒'%(myAlgorithm.passTime))
	population.save()

def GA_Run_All_Model():
	opath = "D:\\sudty\\遗传算法_岩性权重\\3口井权重结果\\精度变化曲线\\"
	fp = open(os.path.join(opath, "spent_time.txt"), 'w')
	fp.close()
	fp=open(os.path.join(opath,"spent_time.txt"),'a')
	print(1)
	for i in range(6):

		problem = MyProblem.MyProblem_SingleAim(i)  # 实例化问题对象

		BestACClist, AVGACCList = [], []
		headline = ""  # 行头
		GA_Models = Models.Get_GA_Model()
		for model_name, model in GA_Models.items():
			"""==============================种群设置==========================="""
			Encodings = 'RI'  # 编码方式，很坑爹，必须用单引号
			NIND = 50  # 种群大小
			Field = ea.crtfld(Encodings, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
			population=ea.Population(Encodings,Field,NIND) #实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
			model=model(problem,population)
			print("======================================================================================保留标签:，modelname",i,model_name)
			headline+=(model_name+"\t")
			# print(model_name,model)
			# print(MyProblem.BestACC_list)
			# continue
			"""===========================算法参数设置=========================="""
			# myAlgorithm=Mysoea_steadyGA_templet(problem,population)#实例化一个算法模板对象
			# myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # 实例化一个算法模板对象
			myAlgorithm = model  # 实例化一个算法模板对象
			myAlgorithm.MAXGEN=10 #最大遗传代数
			# myAlgorithm.mutOper.F = 0.5 # 设置差分进化的变异缩放因子
			# myAlgorithm.recOper.XOVR=0.5 #设置交叉概率
			# myAlgorithm.drawing=1 #设置绘图方式
			"""=====================调用算法模板进行种群进化====================="""
			start_time=time.time()
			[population,obj_trace,var_trace]=myAlgorithm.run() #执行算法模板
			spent_time=time.time()-start_time
			print(spent_time,sep="\t",file=fp)
			# print(MyProblem.BestACC_list)
			BestACClist.append(np.array(MyProblem.BestACC_list))
			AVGACCList.append(np.array(MyProblem.AvgACC_list))
			MyProblem.BestACC_list.clear()
			MyProblem.AvgACC_list.clear()
			# #输出结果
			# best_gen=np.argmax(obj_trace[:,1]) #记录最优种群是在那一代
			# best_Objv=obj_trace[best_gen,1]
			# print('最优的目标函数值为：%s'%(best_Objv))
			# print('最优的决策变量值为：')
			# for i in range(var_trace.shape[1]):
			# 	print(var_trace[best_gen, i])
			# print('有效进化代数：%s'%(obj_trace.shape[0]))
			# print('最优的一代是第%s 代'%(best_gen + 1))
			# print('评价次数：%s'%(myAlgorithm.evalsNum))
			# print('时间已过%s 秒'%(myAlgorithm.passTime))
			# population.save()
		print("\n",file=fp)
		BestACC=np.column_stack(BestACClist)
		AvGACC=np.column_stack(AVGACCList)
		np.savetxt(os.path.join(opath,"保存label为_"+str(i)+"_最好个体精度曲线.txt"),BestACC,fmt="%.04f",header=headline,comments="",delimiter="\t")
		np.savetxt(os.path.join(opath,"保存label为_"+str(i)+"_平均种群精度曲线.txt"),AvGACC,fmt="%.04f",header=headline,comments="",delimiter="\t")
	fp.close()
if __name__=="__main__":
	# GA_Run()
	GA_Run_All_Model()
