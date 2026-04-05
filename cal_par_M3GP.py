import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from Generation_M3GP import generate_tree,tree2expr,expr2func
from NODE import NODE
from copy import deepcopy as dc
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from multiprocessing import Pool

def geno2pheno(genotype,data):
    phenotype=np.zeros([len(data),len(genotype)])
    for i in range(len(genotype)): # 父代
        f=expr2func(tree2expr(genotype[i]),61,1)
        func=eval("lambda x: " + f)
        phenotype[:,i]=func(data)
    return phenotype

def cal_Fitness(genotype,data,label,id=-1,seed=42):
    phenotype = geno2pheno(genotype, data)

    (TrainX, TestX, label1, label2) = train_test_split(phenotype, label, test_size=0.2, random_state=seed)
    TrainY = label1[:, 0]
    TrainR = label1[:, 1:]
    TestY = label2[:, 0]
    TestR = label2[:, 1:]

    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(random_state=seed)
    # 训练模型
    rf_classifier.fit(TrainX, TrainY)
    y_hat = rf_classifier.predict(TestX)

    acc = 0
    for i in range(len(TestY)):
        if TestY[i] == y_hat[i]:
            acc += 1
        elif stats.ranksums(TestR[i, int(y_hat[i] * 30):int((y_hat[i] + 1) * 30)],
                            TestR[i, int(TestY[i] * 30):int((TestY[i] + 1) * 30)])[1] > 0.05:
            acc += 1

    if id!=-1:
        return (id,acc / len(TestY))
    return accuracy_score(TestY, y_hat), acc / len(TestY)


def cal_Fitness_par(Pop,data,label,seed):
    pool = Pool(10)
    results = []

    for i in range(len(Pop)):
        results.append(pool.apply_async(cal_Fitness, (Pop[i],data,label, i,seed)))
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    fitness=[]
    for res in results:
        fitness.append(res.get()[1])
    return fitness