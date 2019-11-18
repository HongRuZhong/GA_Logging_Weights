import geatpy as ea

# def Get_GA_Model(problem, population):
# 	models={}
# 	models["ES_1_plus"]=ea.soea_ES_1_plus_1_templet(problem, population)
# 	models["SGA"]=ea.soea_SGA_templet(problem, population)
# 	models["GGAP_SGA"]=ea.soea_GGAP_SGA_templet(problem, population)
# 	models["EGA"]=ea.soea_EGA_templet(problem, population)
# 	models["SEGA"]=ea.soea_SEGA_templet(problem, population)
# 	models["DE_TargetToBest_1_bin"]=ea.soea_DE_targetToBest_1_bin_templet(problem, population)
# 	models["DE_Best_1_L"]=ea.soea_DE_best_1_L_templet(problem, population)
# 	models["strdGA"]=ea.soea_studGA_templet(problem, population)
# 	models["DE_rand_1_L"]=ea.soea_DE_rand_1_L_templet(problem, population)
# 	models["DE_rand_1_bin"]=ea.soea_DE_rand_1_bin_templet(problem, population)
# 	return models

def Get_GA_Model():
	models={}

	# models["DE_rand_1_L"]=ea.soea_DE_rand_1_L_templet
	models["DE_rand_1_bin"]=ea.soea_DE_rand_1_bin_templet
	return models  #只要这两个最好的
	models["ES_1_plus"]=ea.soea_ES_1_plus_1_templet
	models["SGA"]=ea.soea_SGA_templet
	models["GGAP_SGA"]=ea.soea_GGAP_SGA_templet

	models["EGA"]=ea.soea_EGA_templet
	models["SEGA"]=ea.soea_SEGA_templet
	models["DE_TargetToBest_1_bin"]=ea.soea_DE_targetToBest_1_bin_templet
	models["DE_Best_1_L"]=ea.soea_DE_best_1_L_templet
	models["strdGA"]=ea.soea_studGA_templet
	models["DE_rand_1_L"]=ea.soea_DE_rand_1_L_templet
	models["DE_rand_1_bin"]=ea.soea_DE_rand_1_bin_templet
	return models