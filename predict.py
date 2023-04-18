import torch
import numpy as np
from tqdm import tqdm
from model import Model
from sklearn import metrics
from utils import load_data,calculate

trainset = "/train_fasta"
trainlabel = "/train_label"
testset = "/test_fasta"
testlabel = "/test_label"

# path
Dataset_Path = './data/'
Result_Path = './result/'

# Seed
SEED = 4396
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

def getlabel(fl):
    if fl >=0.5:
        return 1
    else:
        return 0 

def evaluate(model,val_features, val_graphs, val_labels, model_name, dpath, val_nodes):
    
    model.eval()
    preds = torch.zeros(len(val_labels)).cuda()             
    confusion = torch.zeros(2,2).cuda()                   
    root = ["not-root", "root"]    
    f1 = open(Result_Path + "root_prediction_"+model_name+".txt", "w")

    for i in tqdm(range(len(val_labels))):
        with torch.no_grad():
            
            sequence_features = torch.from_numpy(val_features[i])
            sequence_graphs = torch.from_numpy(val_graphs[i])
            sequence_nodes = torch.from_numpy(val_nodes[i])
            labels = torch.from_numpy(np.array([int(float(val_labels[i]))]))

            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)
            sequence_nodes = torch.squeeze(sequence_nodes)

            if torch.cuda.is_available():
                features = sequence_features.cuda()
                graphs = sequence_graphs.cuda()
                nodes = sequence_nodes.cuda()
                y_true = labels.cuda()

            a = y_true.item()
            y_pred = model(features, graphs, nodes)
            new_y_pred = getlabel(y_pred)
            preds[i] = y_pred

            f1.write("prediction: " + str(new_y_pred) + " " + root[new_y_pred] + "\n")
            confusion[a][new_y_pred] += 1                                                               
    f1.close()
    q = preds.cpu()
    labels = []
    f3 = open(dpath + testlabel, "r")
    for j in range(len(val_labels)):                        
        label = f3.readline().strip()                       
        if(int(label[0])==1):                               
            labels.append(1)                                
        else:                                               
            labels.append(0)                                
    rocauc = metrics.roc_auc_score(labels, q)                                                                
    f3.close()
    acc, table = calculate(confusion)                                                                     
    print("acc:", round(acc,4))
    print(" ".ljust(17, ' ') + "MCC    "+ "FSC    " + "Precision  " + "Recall  " + "accuracy  " + "sensitivity  " + "specificity  " + "AUC   ")
    print(root[1].ljust(17, ' ') + str(round(table[6],3)).ljust(7, ' ') + str(round(table[0],3)).ljust(7, ' ') + str(round(table[1],3)).ljust(11, ' ') + \
           str(round(table[2],3)).ljust(8, ' ') + str(round(table[3],3)).ljust(10, ' ') + str(round(table[4],3)).ljust(13, ' ') \
            + str(round(table[5],3)).ljust(13, ' ') + str(round(rocauc,3)).ljust(6, ' '))
    
    return str(round(table[6],3)), str(round(table[0],3)), str(round(table[1],3)), str(round(table[2],3)), str(round(table[3],3)), str(round(table[4],3)), str(round(table[5],3)), str(round(rocauc,3))

def main(model_name, data_path, n2vname):
    
    model = Model()
    model.load_state_dict(torch.load('./model/best_model_'+model_name+'.pkl'))
    if torch.cuda.is_available():
        model.cuda()
    val_features, val_graphs, val_labels, val_nodes = load_data(data_path + testset, data_path + testlabel, Dataset_Path + "pssm/", Dataset_Path + "graph/",Dataset_Path + "n2v" + n2vname)
    mcc, fsc, prec, recall, acc, sensi, speci, auc = evaluate(model,val_features, val_graphs, val_labels, model_name, data_path, val_nodes)
    f = open("predict_result.txt", "a")
    f.write(mcc + " \t" + fsc + " \t" + prec  + " \t" + recall  + " \t" + acc  + " \t" +sensi  + " \t" +  speci  + " \t" +  auc + "\n")
    f.close()

if __name__ == "__main__":
    print("This is predict.py")