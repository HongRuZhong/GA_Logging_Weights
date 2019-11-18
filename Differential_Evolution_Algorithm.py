'''本py是从网上找了用python实现差分进化算法，网址为：https://blog.csdn.net/wang454592297/article/details/80375541'''
import numpy as np
import random

class DE:
    def __init__(self, dim, size, iter_num, x_min, x_max, best_fitness_value=float('Inf'), F = 0.5, CR = 0.8):
        self.F = F
        self.CR = CR
        self.dim = dim  # 维度
        self.size = size  # 总群个数
        self.iter_num = iter_num  # 迭代次数
        self.x_min = x_min
        self.x_max = x_max
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 全局最优解
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.unit_list = [Unit(self.x_min, self.x_max, self.dim) for i in range(self.size)]


    # 变异
    def mutation_fun(self):
        for i in range(self.size):
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.size - 1)  # 随机数范围为[0,size-1]的整数
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
            mutation = self.get_kth_unit(r1).get_pos() + \
                       self.F * (self.get_kth_unit(r2).get_pos() - self.get_kth_unit(r3).get_pos())
            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.x_min <= mutation[j] <= self.x_max:
                    self.get_kth_unit(i).set_mutation(j, mutation[j])
                else:
                    rand_value = self.x_min + random.random()*(self.x_max - self.x_min)
                    self.get_kth_unit(i).set_mutation(j, rand_value)

    # 交叉
    def crossover(self):
        for unit in self.unit_list:
            for j in range(self.dim):
                rand_j = random.randint(0, self.dim - 1)
                rand_float = random.random()
                if rand_float <= self.CR or rand_j == j:
                    unit.set_crossover(j, unit.get_mutation()[j])
                else:
                    unit.set_crossover(j, unit.get_pos()[j])

    # 选择
    def selection(self):
        for unit in self.unit_list:
            new_fitness_value = fit_fun(unit.get_crossover())
            if new_fitness_value < unit.get_fitness_value():
                unit.set_fitness_value(new_fitness_value)
                for i in range(self.dim):
                    unit.set_pos(i, unit.get_crossover()[i])
            if new_fitness_value < self.get_bestFitnessValue():
                self.set_bestFitnessValue(new_fitness_value)
                for j in range(self.dim):
                    self.set_bestPosition(j, unit.get_crossover()[j])

    def update(self):
        for i in range(self.iter_num):
            self.mutation_fun()
            self.crossover()
            self.selection()
            self.fitness_val_list.append(self.get_bestFitnessValue())
        return self.fitness_val_list, self.get_bestPosition()



import matplotlib.pyplot as plt

dim = 2
size = 15
iter_num = 1000
x_max = 10

de = DE(dim, size, iter_num, -x_max, x_max)
fit_var_list, best_pos = de.update()
print("DE最优位置:" + str(best_pos))
print("DE最优解:" + str(fit_var_list[-1]))
plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list, c="G", alpha=0.5)
plt.show()

