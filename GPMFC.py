# filter GP。为一个类回归出一颗树，这个参数不变
import operator, math
from deap import gp, tools, base, creator, algorithms
import numpy as np
import random
from Benchmarks import get_dataset
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
import csv


def GPMFC(seed,data,label,info_savepath):
    random.seed(seed)
    np.random.seed(seed)

    # Parameter setting
    CXPB = 0.6
    MUPB = 0.3
    REPROB = 0.1  # 复制概率
    GEN = 50
    POP_SIZE =1000
    Elitism=1

    (TrainX, TestX, label1, label2) = train_test_split(data, label, test_size=0.2, random_state=seed)
    TrainY = label1[:, 0]


    F_tree=[]
    for c_star in range(10):
        toolbox = base.Toolbox()
        pset = gp.PrimitiveSet("MAIN", 61)

        # define protectedDive function which return their division
        def protectedDiv(left, right):
            if right==0:
                return 0
            else:
                return left / right


        # adding other functions
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)


        # add random constants which is an int type
        pset.addEphemeralConstant("rand", lambda: random.random())
        # rename augument x
        # pset.renameArguments(ARG0='x')

        # creating MinFit
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # creating individual
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # import toolbox
        toolbox = base.Toolbox()
        # resigter expr individual population and compile
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=7)  # 0~3
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def find_interval(y, c, c_star, coverage_ratio=0.99):
            """
            Find the interval [l, u] that covers the specified class c_star
            at the given coverage ratio.

            Parameters:
                y (list or array): A vector of n transformed observations of original features.
                c (list or array): A vector of class labels corresponding to each observation in y.
                c_star (int): The class for which the interval should be found.
                coverage_ratio (float): The coverage ratio (default is 0.99).

            Returns:
                (float, float): The boundaries of the interval (l, u) for class c_star.
            """

            # Step 1: Initialize variables
            outliers = 1 - coverage_ratio
            n_c_star = sum(1 for i in range(len(c)) if c[i] == c_star)  # Number of samples in class c_star

            Left = []  # Lower extreme values
            Right = []  # Upper extreme values

            # Step 2: Iterate through all samples
            for i in range(len(y)):
                if c[i] == c_star:
                    # Update Left boundary
                    if len(Left) > 0 and y[i] < max(Left):
                        if len(Left) >= np.ceil(outliers / 2 * n_c_star):
                            Left.remove(max(Left))
                    Left.append(y[i])

                    # Update Right boundary
                    if len(Right) > 0 and y[i] > min(Right):
                        if len(Right) >= np.ceil(outliers / 2 * n_c_star):
                            Right.remove(min(Right))
                    Right.append(y[i])

            # Step 3: Compute interval boundaries
            l = max(Left) if Left else float('-inf')
            u = min(Right) if Right else float('inf')

            return l, u

        def calculate_fitness(phi,X,Y, c_star):

            # Step 1: Transform the dataset using phi
            y = [phi(*x) for x in X]  # Transformed scalar values
            c = [label for label in Y]  # Class labels

            # Step 2: Find the interval for the target class
            l, u = find_interval(y, c, c_star)

            # Step 3: Initialize fitness
            fitness = 0

            # Step 4: Calculate fitness
            for i in range(len(X)):
                if l <= y[i] <= u:  # If the transformed value is within the interval
                    if c[i] != c_star:  # If it belongs to the wrong class
                        fitness += 1

            return fitness


        def evaluate(individual, X,Y, c_star):

            phi = toolbox.compile(expr=individual)

            return calculate_fitness(phi, X,Y, c_star),


        # register genetic operations(evaluate/selection/mutate/crossover/)
        toolbox.register("evaluate", evaluate,X=TrainX,Y=TrainY,c_star=c_star)

        toolbox.register("select", tools.selTournament, tournsize=7)
        toolbox.register("mate", gp.cxOnePoint)
            # this is expr_mut, if we want to use a GEP, we can mutute the expr at first, then do the expression
        toolbox.register("expr_mut", gp.genFull, min_=2, max_=5)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        # decorating the operator including crossover and mutate, restricting the tree's height and length
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

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

        best_ind = tools.selBest(pop, 1)[0]
        if best_ind.fitness.values[0] == 0:
            # print("-- End of (successful) evolution --")
            # print("Best individual is %s, Best results are %s" % (best_ind, best_ind.fitness.values))
            F_tree.append(best_ind)
            continue

        '''The genetic operations'''
        for g in range(1,GEN):
            # select
            offspring = toolbox.select(pop, len(pop) - int(len(pop) * REPROB))
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
            # print('Evaluated %i individuals' % len(invalid_ind))

            # 选择Elitism个体
            elites = tools.selBest(pop, int(len(pop) * REPROB))
            # update the pop
            pop[:] = elites+offspring


            best_ind = tools.selBest(pop, 1)[0]
            if best_ind.fitness.values[0] == 0:
                F_tree.append(best_ind)
                # print("-- End of (successful) evolution --")
                # print("Best individual is %s, Best results are %s" % (best_ind, best_ind.fitness.values))
                break

    features=np.zeros([len(data),10])
    for i,tree in enumerate(F_tree):
        func=toolbox.compile(expr=tree)
        for j in range(len(data)):
            features[j, i] = func(*data[j, :])
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

    print(seed,acc / len(TestY))

if __name__ == '__main__':
    data, label = get_dataset(0)
    info_savepath = r'D:\Pythonnnnnn\python\FeatureConstruction\EXP\comparison\GPWFC\BBOB.csv'
    seed=42
    GPMFC(seed, data, label, info_savepath)
