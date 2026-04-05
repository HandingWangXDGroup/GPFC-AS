import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
from Generation import generate_tree,tree2expr,expr2func,crossover,mutation,mutation1,tree_edit_distance
from NODE import NODE
from copy import deepcopy as dc
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
from keras import initializers
from keras import regularizers
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

def geno2pheno(genotype,data):
    phenotype=np.zeros([len(data),len(genotype)])
    for i in range(len(genotype)): # 父代
        f=expr2func(tree2expr(genotype[i]),61,1)
        func=eval("lambda x: " + f)
        phenotype[:,i]=func(data)
    return phenotype

def cal_Fitness(genotype,data,label,id=-1,seed=42,Model=RandomForestClassifier):
    phenotype = geno2pheno(genotype, data)

    (TrainX, TestX, label1, label2) = train_test_split(phenotype, label, test_size=0.2, random_state=seed)
    TrainY = label1[:, 0]
    TrainR = label1[:, 1:]
    TestY = label2[:, 0]
    TestR = label2[:, 1:]

    # 创建随机森林分类器
    try:
        model = Model(random_state=seed)
    except:
        model = Model()
    # 训练模型
    model.fit(TrainX, TrainY)
    y_hat = model.predict(TestX)

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


def cal_Fitness_par(Pop,data,label,seed,Model=RandomForestClassifier):
    pool = Pool(10)
    results = []

    for i in range(len(Pop)):
        results.append(pool.apply_async(cal_Fitness, (Pop[i],data,label, i,seed,Model)))
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    fitness=[]
    for res in results:
        fitness.append(res.get()[1])
    return fitness


def cal_Fitness_FCN(genotype,data,label,seed=42):
    phenotype = geno2pheno(genotype, data)

    (TrainX, TestX, label1, label2) = train_test_split(phenotype, label, test_size=0.2, random_state=seed)
    TrainY = label1[:, 0]
    TrainR = label1[:, 1:]
    TestY = label2[:, 0]
    TestR = label2[:, 1:]

    # 模型结构
    model = Sequential()
    model.add(Dense(64, input_shape=(len(TrainX[0]),), activation="relu",
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=seed)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu",
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=seed)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu",
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=seed)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu",
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=seed)))
    model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(label[:, 0])), activation='sigmoid',
                    kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=seed)))

    # 指定参数


    opt = Adam(learning_rate=0.01, decay=0.01 / 20)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    lb = LabelBinarizer()
    trainY = lb.fit_transform(TrainY)
    testY = lb.transform(TestY)
    # 训练模型

    H = model.fit(TrainX, trainY, validation_data=(TestX, testY),
                  epochs=20, batch_size=64, verbose=0)

    # 预测结果分析
    prediction = model.predict(TestX)
    y = np.argmax(testY, axis=1)
    y_hat = np.argmax(prediction, axis=1)

    acc = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            acc += 1
        elif stats.ranksums(TestR[i, int(y_hat[i] * 30):int((y_hat[i] + 1) * 30)],
                            TestR[i, int(y[i] * 30):int((y[i] + 1) * 30)])[1] > 0.05:
            acc += 1
    return 0,acc / len(y)

def cal_Fitness_par_FCN(Pop,data,label,seed):
    pool = Pool(5)
    results = []

    for i in range(len(Pop)):
        results.append(pool.apply_async(cal_Fitness_FCN, (Pop[i],data,label,seed)))
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    fitness=[]
    for res in results:
        fitness.append(res.get()[1])
    return fitness
