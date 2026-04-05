# 先写fitness然后搭框架
import operator, math
from deap import gp, tools, base, creator, algorithms
import numpy as np
import random
from Benchmarks import get_dataset
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
import csv


def CDFC(seed,data,label,info_savepath):
    random.seed(seed)
    np.random.seed(seed)
    savepath=info_savepath+r'{}.csv'.format(seed)

    # Parameter setting
    CXPB = 0.8
    MUPB = 0.2
    GEN = 50
    POP_SIZE =40
    r=2
    nTree=10*r
    alpha=0.8
    Elitism=1
    (TrainX, TestX, label1, label2) = train_test_split(data, label, test_size=0.2, random_state=seed)
    TrainY = label1[:, 0]

    # 计算无条件熵
    unique, counts = np.unique(TrainY, return_counts=True)
    probabilities = counts / len(TrainY)
    # 计算熵 H(Y) = -sum(p * log2(p))
    Hclass = -np.sum(probabilities * np.log2(probabilities))
    # print(unique)
    # print(counts)
    # print(Hclass)

    # 计算原始特征的类别相关性
    foc=[]
    for c in range(10):
        rel_fc=[]
        group_c=TrainY==c
        group_nc=TrainY!=c
        for j in range(len(data[0])):
            t_stat, p_value=stats.ttest_ind(TrainX[group_c,j], TrainX[group_nc,j], equal_var=False)
            if p_value >= 0.05:
                rel_fc.append(0)
            elif p_value==0:
                rel_fc.append(np.inf)
            elif np.isnan(p_value):
                rel_fc.append(0)
            else:
                rel_fc.append(abs(t_stat) / p_value)# 计算相关性度量
        # 找出每一类所用特征的索引,并进行扩展
        sorted_indices = np.argsort(rel_fc)[::-1]
        top_half_indices = sorted_indices[:len(rel_fc)//2]
        if len(rel_fc)%2:
            top_indices=np.array(top_half_indices.tolist() * 2+[top_half_indices[0]])
        else:
            top_indices = np.array(top_half_indices.tolist() * 2)
        foc.append(top_indices)
    # print(foc)
    toolbox = base.Toolbox()
    pset = gp.PrimitiveSet("MAIN", len(data[0]))

    def if_then_else(a,b,c):
        if a>0:
            return b
        else:
            return c

    def max(a,b):
        if a>b:
            return a
        else:
            return b


    # adding other functions
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(if_then_else, 3)

    # creating MinFit
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creating individual
    creator.create("Tree", gp.PrimitiveTree, fitness=creator.FitnessMax)
    creator.create("MultiTree", list, fitness=creator.FitnessMax)
    # import toolbox
    toolbox = base.Toolbox()
    # resigter expr individual population and compile
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=4, max_=7)  # 0~3
    toolbox.register("Tree", tools.initIterate, creator.Tree,toolbox.expr)
    toolbox.register("ind", tools.initRepeat, list,toolbox.Tree,n=nTree)
    toolbox.register("Individual", tools.initIterate, creator.MultiTree,toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
    toolbox.register("compile", gp.compile, pset=pset)



    # 计算类间距离Db
    def calculate_distance(S, labels):
        """
        计算类间距离Db，S是数据集，feature_matrix是特征矩阵，labels是类别标签
        """
        num_samples = len(S)
        db = 0.0
        dw = 0.0
        distance_matrix = pdist(S)
        distance_matrix_square = squareform(distance_matrix)
        for i in range(num_samples):
            min_distance = np.min(distance_matrix_square[i, labels != labels[i]])
            max_distance = np.max(distance_matrix_square[i, labels == labels[i]])
            db += min_distance
            dw += max_distance

        db /= num_samples
        dw /= num_samples

        # 计算最终的距离度量
        distance = 1 / (1 + np.exp(-5 * (db - dw)))
        return distance



    # 计算条件熵 H(y | f)
    def conditional_entropy(feature, target):
        """
        计算连续特征离散化后的条件熵 H(y | f)

        feature: 连续特征数据 (numpy 数组)
        target: 目标变量数据 (numpy 数组)
        """
        # 获取特征的唯一值（即进行离散化的区间数）
        unique_values = np.unique(feature)

        cond_entropy = 0
        total_samples = len(feature)

        for value in unique_values:
            # 获取特征为 value 的子集
            subset_target = target[feature == value]

            # 计算目标变量的类别概率
            _, counts = np.unique(subset_target, return_counts=True)
            probs = counts / len(subset_target)

            # 计算该子集的熵
            subset_entropy = -np.sum(probs * np.log2(probs + 1e-9))

            # 加权条件熵
            cond_entropy += (len(subset_target) / total_samples) * subset_entropy

        return cond_entropy

    def evaluate(individual,data,label,Hclass,foc):
        features = np.zeros([len(data), nTree])
        for i,tree in enumerate(individual):
            func = toolbox.compile(expr=tree)
            feature_of_class=data[:,foc[i//r]]
            for j in range(len(data)):
                features[j, i] = func(*feature_of_class[j, :])

        features[features > 10 ** 8] = 10 ** 8
        features[features < -10 ** 8] = -10 ** 8
        features[np.isnan(features)] = 0
        features[np.isinf(features)] = 0

        # 计算条件熵
        Hclass_f=[]
        for i in range(nTree):
            feature=features[:,i]
            # 离散化特征
            bins = np.linspace(np.min(feature), np.max(feature), 100)  # 3 个区间
            feature_discretized = np.digitize(feature, bins)  # 将连续特征离散化为区间编号
            Hclass_f.append(conditional_entropy(feature_discretized, label))

        AvgIG=Hclass-(np.sum(Hclass_f)+np.min(Hclass_f))/(nTree+1)

        distance= calculate_distance(features / 10 ** 8, label)

        indSize=0
        for i in range(nTree):
            indSize+=len(individual[i])

        fit=alpha*AvgIG+(1-alpha)*distance-indSize*10**-7
        return fit,


    # register genetic operations(evaluate/selection/mutate/crossover/)
    toolbox.register("evaluate", evaluate,data=TrainX,label=TrainY,Hclass=Hclass,foc=foc)

    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
        # this is expr_mut, if we want to use a GEP, we can mutute the expr at first, then do the expression
    toolbox.register("expr_mut", gp.genFull, min_=2, max_=5)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # decorating the operator including crossover and mutate, restricting the tree's height and length
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))


    # initializing population
    pop = toolbox.population(n=POP_SIZE)

    # '''start evolution'''
    # print('Start evolution')
    # evaluating the fitness
    fitnesses = list(map(toolbox.evaluate, pop))
    # assign fitness values
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # print('Evaluated %i individuals' % len(pop))

    '''The genetic operations'''
    for g in range(1, GEN):
        if seed ==0:
            print(g)
        # select
        offspring = toolbox.select(pop, len(pop) - Elitism)
        offspring = list(map(toolbox.clone, offspring))
        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                cind=np.random.randint(nTree)
                toolbox.mate(child1[cind], child2[cind])

                del child1.fitness.values
                del child2.fitness.values

        # mutation
        for mutant in offspring:
            if random.random() < MUPB:
                mind = np.random.randint(nTree)
                toolbox.mutate(mutant[mind])
                del mutant.fitness.values

        # evaluate the invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # print('Evaluated %i individuals' % len(invalid_ind))

        # 选择Elitism个体
        elites = sorted(pop, key=lambda ind: ind.fitness.values, reverse=True)[:Elitism]
        # update the pop
        pop[:] = elites + offspring
    #
    #     # 记录
    #     csvfile = open(savepath, 'a+', newline='')
    #     csvwriter = csv.writer(csvfile)
    #     best_ind = tools.selBest(pop, 1)[0]
    #     bestFitness = best_ind.fitness.values[0]
    #     for ind in pop:
    #         log_info = []
    #         log_info.append(g)
    #         log_info.append(bestFitness)
    #         log_info.append(ind.fitness.values[0])
    #
    #         csvwriter.writerow(log_info)
    #         csvfile.flush()
    #     csvfile.close()

        # statistics
        # stat = tools.Statistics(key=lambda ind: ind.fitness.values)
        # record = stat.compile(pop)
        # stat.register("avg", np.mean, axis=0)
        # stat.register("min", np.min, axis=0)
        # stat.register("max", np.max, axis=0)
        # record = stat.compile(pop)
        # logbook = tools.Logbook()
        # logbook.record(gen=g, evals=30, **record)
        # logbook.header = "gen", "avg", "min", "max"
        # print(logbook)
    #
    # print("-- End of (successful) evolution --")
    # best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, Best results are %s" % (best_ind, best_ind.fitness.values))


    features=np.zeros([len(data),nTree])
    best_ind = tools.selBest(pop, 1)[0]
    for i,tree in enumerate(best_ind):
        func=toolbox.compile(expr=tree)
        feature_of_class = data[:, foc[i // r]]
        for j in range(len(data)):
            features[j, i] = func(*feature_of_class[j, :])
    # 异常数值处理
    features[features > 10 ** 8] = 10 ** 8
    features[features < -10 ** 8] = -10 ** 8
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 0

    (TrainX, TestX, label1, label2) = train_test_split(features, label, test_size=0.2, random_state=seed)

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

    # save info
    csvfile = open(info_savepath, 'a+', newline='')
    csvwriter = csv.writer(csvfile)
    log_info = [seed,acc / len(TestY)]
    csvwriter.writerow(log_info)
    csvfile.flush()
    csvfile.close()



if __name__ == '__main__':
    data, label = get_dataset(0)
    info_savepath = r'D:\Pythonnnnnn\python\FeatureConstruction\EXP\comparison\CDFC\BBOB.csv'
    seed = 0
    CDFC(seed, data, label, info_savepath)