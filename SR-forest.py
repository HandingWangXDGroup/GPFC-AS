from Benchmarks import get_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evolutionary_forest.forest import EvolutionaryForestRegressor, EvolutionaryForestClassifier
import numpy as np
import random


def SR-Forest(seed,data,label,info_savepath):
    random.seed(seed)
    np.random.seed(seed)

    (TrainX, TestX, label1, label2) = train_test_split(data, label, test_size=0.2, random_state=seed)
    TrainY = label1[:, 0]
    TrainR = label1[:, 1:]
    TestY = label2[:, 0]
    TestR = label2[:, 1:]

    est = EvolutionaryForestClassifier(max_height=8, normalize=True, select='AutomaticLexicase', boost_size=100,
                                      basic_primitives='sin-cos', mutation_scheme='EDA-Terminal-PM',
                                      semantic_diversity='GreedySelection-Resampling', initial_tree_size='2-6',
                                      cross_pb=0.9, mutation_pb=0.1, gene_num=20, n_gen=50,score_func='CrossEntropy',
                                      n_pop=40, base_learner='DT-LR',verbose=True)

    est.fit(TrainX, TrainY)
    y_hat = est.predict(TestX)

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
    info_savepath = r'D:\Pythonnnnnn\python\FeatureConstruction\EXP\comparison\SR-Forest\BBOB.csv'

    process_list = []
    for i in range(30):  # 开启5个子进程执行fun1函数
        p = Process(target=SR-Forest, args=(i, data, label, info_savepath))  # 实例化进程对象
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
