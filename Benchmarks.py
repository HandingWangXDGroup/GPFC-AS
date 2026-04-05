import pandas as pd
import numpy as np

def get_dataset(ID):
    # ID: 0BBOB,1Affine,3GPB,4RGF
    if ID==0:
        # BBOB
        BBOB_ela = np.zeros([1, 61])
        BBOB_rt = np.zeros([1, 300])
        for i in range(24):
            BBOB_ela = np.append(BBOB_ela,
                                 np.loadtxt(open("D:\Dataset\Mywork\DataAll\BBOB\\bbob-{}.csv".format(i + 1), "r"),
                                            delimiter=",")[:, 600:], axis=0)
            BBOB_rt = np.append(BBOB_rt,
                                np.loadtxt(open("D:\Dataset\Mywork\DataAll\BBOB\\bbob-{}.csv".format(i + 1), "r"),
                                           delimiter=",")[:, :300], axis=0)

        BBOB_ela, BBOB_rt = data_screening(BBOB_ela, BBOB_rt)
        BBOB_ert, BBOB_label = labeling(BBOB_rt)
        return BBOB_ela, BBOB_label

    elif ID==1:
        # BBOB_Affine
        Affine= np.loadtxt(open(r"D:\Dataset\Mywork\DataAll\Affine\affine.csv", "r"), delimiter=",")
        Affine_ela = Affine[:, 600:]
        Affine_rt = Affine[:, :300]

        Affine_ela, Affine_rt = data_screening(Affine_ela, Affine_rt)
        Affine_ert, Affine_label = labeling(Affine_rt)
        return Affine_ela, Affine_label
    elif ID == 2:
        # GPB
        GPB_ela = np.zeros([1, 61])
        GPB_rt = np.zeros([1, 300])
        for i in range(5):
            GPB_ela = np.append(GPB_ela, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfA-{}.csv".format(i), "r"),
                                                    delimiter=",")[:, 600:], axis=0)
            GPB_rt = np.append(GPB_rt, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfA-{}.csv".format(i), "r"),
                                                  delimiter=",")[:, :300], axis=0)
        for i in range(5):
            GPB_ela = np.append(GPB_ela, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfB-{}.csv".format(i), "r"),
                                                    delimiter=",")[:, 600:], axis=0)
            GPB_rt = np.append(GPB_rt, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfB-{}.csv".format(i), "r"),
                                                  delimiter=",")[:, :300], axis=0)

        GPB_ela, GPB_rt = data_screening(GPB_ela, GPB_rt)
        GPB_ert, GPB_label = labeling(GPB_rt)
        return GPB_ela,GPB_label

    elif ID == 3:
        # RGF
        RGF_ela = np.zeros([1, 61])
        RGF_rt = np.zeros([1, 300])
        for i in range(10):
            RGF_ela = np.append(RGF_ela, (pd.read_csv(
                "D:\Dataset\Mywork\DataAll\RGF_Dataset\Dataset_ELA\ela-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:3000, 1:], axis=0)
            RGF_rt = np.append(RGF_rt, (np.loadtxt(
                open("D:\Dataset\Mywork\DataAll\RGF_Dataset\Dataset_Best\Best-{}.csv".format(i + 1), "r"),
                delimiter=","))[:3000, 1:], axis=0)
        # 筛选数据
        RGF_ela, RGF_rt = data_screening(RGF_ela, RGF_rt)
        RGF_ert, RGF_label = labeling(RGF_rt)
        return RGF_ela, RGF_label


def labeling(lRT):
    lERT = cal_mean(lRT)
    y = np.argmin(lERT, axis=1)
    label = np.zeros([len(y), 301])
    label[:, 0] = y
    label[:, 1:] = lRT
    return lERT, label

def data_screening(ela,rt=0):
    a=np.sum(ela,axis=1)!=0
    b=np.sum(np.isnan(ela),axis=1)==0
    c=np.sum(np.isinf(ela),axis=1)==0
    d=np.sum(np.abs(ela)>10**8,axis=1)==0
    e = np.sum(np.isnan(rt), axis=1) == 0
    f = np.sum(np.isinf(np.abs(rt)), axis=1) == 0
    ind=a & b & c & d&e&f
    return ela[ind,:],rt[ind,:]

def cal_mean(lRT):
    lERT=np.zeros([len(lRT),10])
    for i in range(len(lRT)):
        for j in range(10):
            lERT[i,j]=np.mean(lRT[i,j*30:(j+1)*30])
    return lERT

if __name__=='__main__':
    d,l=get_dataset(3)
    print()


