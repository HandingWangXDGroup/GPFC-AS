# 单树GP。这个算法是用树的所有子树组成的特征 t、elisit、nPOP、Gen
import operator, math
from deap import gp, tools, base, creator, algorithms
import numpy as np
import random
from Benchmarks import get_dataset
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
import csv


def GPWFC(seed,data,label,info_savepath):
    random.seed(seed)
    np.random.seed(seed)
    savepath=info_savepath+r'{}.csv'.format(seed)

    toolbox = base.Toolbox()
    pset = gp.PrimitiveSet("MAIN", 61)

    # define protectedDive function which return their division
    def protectedDiv(left, right):
        if right==0:
            return 0
        else:
            return left / right

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

    def min(a,b):
        if a>b:
            return b
        else:
            return a

    # adding other functions
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    pset.addPrimitive(if_then_else, 3)
    pset.addPrimitive(math.tanh, 1)

    # add random constants which is an int type
    pset.addEphemeralConstant("rand", lambda: random.random()*20-10)
    # rename augument x
    # pset.renameArguments(ARG0='x')

    # creating MinFit
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creating individual
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # import toolbox
    toolbox = base.Toolbox()
    # resigter expr individual population and compile
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=5, max_=6)  # 0~3
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    aaa=np.random.random([61])
    # define evaluating function
    def cal_Fitness(data,label,seed=42):
        return 0
        (TrainX, TestX, label1, label2) = train_test_split(data, label, test_size=0.2, random_state=seed)

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

        return acc / len(TestY)

    def evaluate(individual,data,label):
        #计算量巨大无比。。。。。
        features = np.zeros([len(data), len(individual)])
        print(len(individual))
        for i in range(len(individual)):
            ind = individual.searchSubtree(i)

            subtree = individual[ind]
            subtree = creator.Individual(subtree)
            func = toolbox.compile(expr=subtree)
            for j in range(len(data)):
                features[j,i]=func(*data[j,:])
        features[features>10**8]=10**8
        features[features < -10 ** 8] = -10 ** 8
        features[np.isnan(features)]=0
        features[np.isinf(features)]=0

        return cal_Fitness(features,label,seed),


    # register genetic operations(evaluate/selection/mutate/crossover/)
    toolbox.register("evaluate", evaluate,data=data,label=label)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint) \
        # this is expr_mut, if we want to use a GEP, we can mutute the expr at first, then do the expression
    toolbox.register("expr_mut", gp.genFull, min_=2, max_=5)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # decorating the operator including crossover and mutate, restricting the tree's height and length
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))


    # Parameter setting
    CXPB = 0.8
    MUPB = 0.19
    GEN = 20
    POP_SIZE =100
    Elitism=1
    # initializing population
    pop = toolbox.population(n=POP_SIZE)

    # '''start evolution'''
    print('Start evolution')
    # evaluating the fitness
    fitnesses = list(map(toolbox.evaluate, pop))
    # assign fitness values
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print('Evaluated %i individuals' % len(pop))
    '''The genetic operations'''
    for g in range(1,GEN):
        # select
        offspring = toolbox.select(pop, len(pop) - Elitism)
        offspring = list(map(toolbox.clone, offspring))
        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        # mutation
        for mutant in offspring:
            if random.random() < MUPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate the invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print('Evaluated %i individuals' % len(invalid_ind))

        # 选择Elitism个体
        elites = sorted(pop, key=lambda ind: ind.fitness.values,reverse=True)[:Elitism]
        # update the pop
        pop[:] = elites+offspring

        # 记录
        # csvfile = open(savepath, 'a+', newline='')
        # csvwriter = csv.writer(csvfile)
        # bestFitness = elites[0].fitness.values[0]
        # for ind in pop:
        #     log_info = []
        #     log_info.append(g)
        #     log_info.append(bestFitness)
        #     log_info.append(ind.fitness.values[0])
        #
        #     csvwriter.writerow(log_info)
        #     csvfile.flush()
        # csvfile.close()

    #     # statistics
    #     stat = tools.Statistics(key=lambda ind: ind.fitness.values)
    #     record = stat.compile(pop)
    #     stat.register("avg", np.mean, axis=0)
    #     stat.register("min", np.min, axis=0)
    #     stat.register("max", np.max, axis=0)
    #     record = stat.compile(pop)
    #     logbook = tools.Logbook()
    #     logbook.record(gen=g, evals=30, **record)
    #     logbook.header = "gen", "avg", "min", "max"
    #     print(logbook)
    #
    # print("-- End of (successful) evolution --")
    #
    # best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, Best results are %s" % (best_ind, best_ind.fitness.values))


if __name__ =='__main__':
    data, label = get_dataset(0)
    info_savepath = r''
    seed = 42
    GPWFC(seed, data, label, info_savepath)

