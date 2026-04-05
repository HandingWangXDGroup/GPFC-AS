# nPop,Gen,
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from NODE import NODE
from copy import deepcopy as dc
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import accuracy_score
import time
from cal_par_M3GP import cal_Fitness_par,cal_Fitness
import csv
from Benchmarks import get_dataset
from Generation_M3GP import generate_tree,tree2expr,expr2func,crossover,mutation


def M3GP(seed,data,label,info_savepath):
    def geno2pheno(genotype):
        phenotype=np.zeros([len(data),len(genotype)])
        for i in range(len(genotype)): # 父代
            f=expr2func(tree2expr(genotype[i]),61,1)
            func=eval("lambda x: " + f)
            phenotype[:,i]=func(data)
        return phenotype


    def Cal_Fitness(genotype, id=-1, seed=42):
        return cal_Fitness(genotype, data, label, id, seed)


    def Cal_Fitness_Par(Pop, seed=42):
        return cal_Fitness_par(Pop, data, label, seed)


    def Validate_tree(individual):
        for i in range(len(individual)):
            pheno = geno2pheno([individual[i]])
            a = np.abs(pheno) > 10 ** 8
            b = np.isnan(pheno)
            c = np.isinf(pheno)
            d = a + b + c
            while np.sum(d) > 0:
                individual[i] = generate_tree(2, 5)
                pheno = geno2pheno([individual[i]])
                a = np.abs(pheno) > 10 ** 8
                b = np.isnan(pheno)
                c = np.isinf(pheno)
                d = a + b + c


    def add_tree(halfandfhalf=0):
        if halfandfhalf:
            genoTree = generate_tree(deep_tree_min, deep_tree_max, np.random.randint(2))
            pheno = geno2pheno([genoTree])
            a = np.abs(pheno) > 10 ** 8
            b = np.isnan(pheno)
            c = np.isinf(pheno)
            d = a + b + c
            while np.sum(d) > 0:
                genoTree = generate_tree(deep_tree_min, deep_tree_max, np.random.randint(2))
                pheno = geno2pheno([genoTree])
                a = np.abs(pheno) > 10 ** 8
                b = np.isnan(pheno)
                c = np.isinf(pheno)
                d = a + b + c
        else:
            genoTree = generate_tree(deep_tree_min, deep_tree_max)
            pheno = geno2pheno([genoTree])
            a = np.abs(pheno) > 10 ** 8
            b = np.isnan(pheno)
            c = np.isinf(pheno)
            d = a + b + c
            while np.sum(d) > 0:
                genoTree = generate_tree(deep_tree_min, deep_tree_max)
                pheno = geno2pheno([genoTree])
                a = np.abs(pheno) > 10 ** 8
                b = np.isnan(pheno)
                c = np.isinf(pheno)
                d = a + b + c
        return genoTree


    def log(savepath, it, Pop, fitness, bestFitness):
        csvfile = open(savepath, 'a+', newline='')
        csvwriter = csv.writer(csvfile)

        for i in range(len(Pop)):
            log_info = []
            log_info.append(it)
            log_info.append(bestFitness)
            log_info.append(fitness[i])
            trees = []
            for j in range(len(Pop[i])):
                trees.append(tree2expr(Pop[i][j]))
            log_info.append('{}'.format(trees))

            csvwriter.writerow(log_info)
            csvfile.flush()
        csvfile.close()

    # 一些超参数
    nPop=20
    MaxIt=100 # 最大迭代次数
    Rcross=0.5
    Rmut=0.5
    n_tournament=2
    deep_tree_min=6
    deep_tree_max=6
    savepath = info_savepath + r'{}.csv'.format(seed)

    np.random.seed(seed)
    # 初始化种群
    Pop = []

    for i in range(nPop):
        Pop.append([])
        Pop[i].append(add_tree())

    fitness=[]
    for i in range(nPop):
        _,acc=Cal_Fitness(Pop[i], seed=seed)
        fitness.append(acc)

    bestFitness = np.max(fitness)
    bestPop = dc(Pop[np.argmax(fitness)])
    his_best = [bestFitness]

    log(savepath, 0, Pop, fitness, bestFitness)

    # 开始迭代
    for it in range(1, MaxIt):
        print(it, bestFitness)

        # crossover
        offspring = []
        ind = np.arange(nPop)
        np.random.shuffle(ind)

        for i in range(0, nPop, 2):
            parent1 = dc(Pop[ind[i]])
            parent2 = dc(Pop[ind[i + 1]])
            if np.random.random() < Rcross:
                if np.random.random() < 0.5:
                    ind_c1 = np.random.randint(len(parent1))
                    ind_c2 = np.random.randint(len(parent2))
                    crossover(parent1[ind_c1], parent2[ind_c2])
                else:
                    ind1 = np.random.randint(len(parent1))
                    ind2 = np.random.randint(len(parent2))
                    temp = dc(parent1[ind1])
                    parent1[ind1] = dc(parent2[ind2])
                    parent2[ind2] = dc(temp)
            offspring.append(parent1)
            offspring.append(parent2)

        # mutation
        for i in range(nPop):
            if np.random.random() < Rmut:
                if np.random.random() < 0.33333 and len(offspring[i])>1:
                    offspring[i].pop(np.random.randint(len(offspring[i])))
                elif np.random.random() < 0.66666:
                    offspring[i].append(add_tree())
                else:
                    mutation(offspring[i][np.random.randint(len(offspring[i]))])
                    Validate_tree(offspring[i])


        ofitness = []
        for i in range(nPop):
            _, acc = Cal_Fitness(offspring[i],seed=seed)
            ofitness.append(acc)

        allPop = dc(Pop + offspring)
        allfitness = fitness + ofitness

        if np.max(allfitness) > bestFitness:
            bestFitness = np.max(allfitness)
            bestPop = dc(allPop[np.argmax(allfitness)])
            his_best.append(bestFitness)

        Pop = []
        fitness = []
        # n元锦标赛
        for i in range(nPop):
            tournament = np.random.choice(np.arange(len(allPop)), n_tournament, replace=False)
            tournament_fitness = []
            for j in range(n_tournament):
                tournament_fitness.append(allfitness[tournament[j]])
            Pop.append(allPop[tournament[np.argmax(tournament_fitness)]])
            fitness.append(allfitness[tournament[np.argmax(tournament_fitness)]])

        #elitism
        Pop[0]=dc(bestPop)
        fitness[0]=bestFitness

        log(savepath, it, Pop, fitness, bestFitness)

if __name__ == '__main__':
    data, label = get_dataset(0)
    info_savepath = r'D:\Pythonnnnnn\python\FeatureConstruction\EXP\comparison\M3GP\BBOB\\'

    process_list = []
    for i in range(30):  # 开启5个子进程执行fun1函数
        p = Process(target=CDFC, args=(i, data, label, info_savepath))  # 实例化进程对象
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
