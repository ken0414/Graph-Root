from Bio import SeqIO
from math import sqrt
import numpy as np

def convertSampleToBlosum62(seq):
    
    """
    Convert a seq to feature matrix of BLOSUM62 values: 20 * 20
    """
    
    letterDict = {}
    letterDict["A"] = [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0]
    letterDict["R"] = [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3]
    letterDict["N"] = [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3]
    letterDict["D"] = [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3]
    letterDict["C"] = [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1]
    letterDict["Q"] = [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2]
    letterDict["E"] = [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2]
    letterDict["G"] = [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3]
    letterDict["H"] = [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3]
    letterDict["I"] = [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3]
    letterDict["L"] = [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1]
    letterDict["K"] = [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2]
    letterDict["M"] = [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1]
    letterDict["F"] = [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1]
    letterDict["P"] = [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2]
    letterDict["S"] = [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2]
    letterDict["T"] = [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0]
    letterDict["W"] = [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3]
    letterDict["Y"] = [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1]
    letterDict["V"] = [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]
    AACategoryLen = 20  
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr

def readPSSM(pssmfile):
    
    """
    read pssm file and return a feature matrix: length * 20
    """
    
    pssm = []
    with open(pssmfile, 'r') as f:
        count = 0
        for eachline in f:
            count += 1
            if count <= 3:
                continue
            if not len(eachline.strip()):
                break
            line = eachline.split()
            pssm.append(line[2: 22])  
    return np.array(pssm)

def readNode2Vec(node2vecfile):
    
    """
    load node2vec file and return a dictionary and the dimension of node2vec feature
    """
    
    f = open(node2vecfile, encoding='utf-8')
    num, dim = f.readline().split()
    node2vec_dic = {}
    for i in range(int(num)):
        l = f.readline().split()
        node2vec_dic[l[0]] = list(map(float, l[1:]))
    return node2vec_dic, dim

def readIPR(iprfile):

    """
    load ipr file and return a dictionary
    """

    f = open(iprfile, encoding='utf-8')
    ipr_dic = {}
    for line in f.readlines():
        line = line.split()
        ipr_dic[line[0]] = np.array(line[1:], dtype=float)
    return ipr_dic

def load_data(seq_file, labelfile, pssmdir, graphdir, nodedir):
    
    labels = []
    features = []
    graphs = []
    nodes = []
    n2v, dim = readNode2Vec(nodedir)
    ipr = readIPR('./data/ipr/ipr_feature.txt')
    dim = int(dim)

    f = open(labelfile, "r")
    num = 0
    print("Load data.")
    for seq_record in list(SeqIO.parse(seq_file, "fasta")):

        pssmfile= pssmdir + str(seq_record.id) +"_pssm.txt"
        pssm = readPSSM(pssmfile)
        pssm = pssm.astype(float)
        Blosum62 = convertSampleToBlosum62(seq_record.seq)
        feature = np.concatenate((Blosum62, pssm), axis=1)
        features.append(feature)
        
        label = f.readline().strip()                               
        labels.append(label)
        
        graph = np.load(graphdir + seq_record.id + ".npy", allow_pickle= True)
        Xgraph_row=np.size(graph,0) 
        for j in range(Xgraph_row):
            for i in range(j+1):
                graph[j][i] = 0

        graph_f=graph.flatten()
        graph_f_s = np.sort(graph_f)[::-1]   
        l_3 = graph_f_s[3 * Xgraph_row]  
        for j in range(Xgraph_row):
            for i in range(j+1, Xgraph_row):
                if graph[j][i] > l_3:
                    graph[j][i] = 1
                else:
                    graph[j][i] = 0
        for j in range(Xgraph_row):
            for i in range(j+1):
                graph[j][i] = graph[i][j]
                graph[j][j] = 1
        graphs.append(graph)         

        if str(seq_record.id) not in n2v:
            n2v_list = [0] * dim
            print(str(seq_record.id) + " do not have node2vec file")
        else:
            n2v_list = n2v[str(seq_record.id)]

        if str(seq_record.id) not in ipr:
            ipr_list = [0] * 50000
            print(str(seq_record.id) + " do not have ipr file")
        else:
            ipr_list = ipr[str(seq_record.id)].tolist()

        nodes.append(n2v_list + ipr_list)
        num += 1
        if(num % 500 == 0):
            print("load " + str(num) + " sequences")

    f.close()
    return features, graphs, labels, np.array(nodes)


def calculate(a):                               
    y_pred = a.cpu().numpy()
    table = np.zeros(7)
    pt = 0
    num = 0
    TP = y_pred[1, 1]
    pt += TP
    FN = y_pred[1, 0]
    FP = y_pred[0, 1]
    TN = y_pred[0, 0]
    pt += TN
    sensitivity = round( TP / (TP + FN), 3) if TP + FN != 0 else 0.
    specificity = round( TN / (TN + FP), 3) if TN + FP != 0 else 0.
    accuracy = round( (sensitivity + specificity) / 2, 3)
    Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
    Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
    mcc_num = round(((TP * TN) - (FP * FN)) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)), 3) if ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) != 0 else 0.
    F_score = round((TP * 2) / (2 * TP + FP + FN), 3) if (2 * TP + FP + FN) != 0 else 0.
    table[0] = F_score
    table[1] = Precision
    table[2] = Recall
    table[3] = accuracy
    table[4] = sensitivity
    table[5] = specificity
    table[6] = mcc_num
    num = np.sum(y_pred)
    return pt/num, table
