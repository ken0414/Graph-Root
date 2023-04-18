import pandas as pd
import os
import train
import predict
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="none", type=str)     # cross-validation or independent test
parser.add_argument('--n2v', default="none", type=str)      # node2vec file name
parser.add_argument('--run', default= 1, type=str)          # run times
args = parser.parse_args()

out_test = "out_test"
positive = "data/positive_example.csv"          # modify if you want to use different dataset
negative = "data/negative_example.csv"
dfpositive = pd.read_csv(positive)
dfnegative = pd.read_csv(negative)

runtimes = int(args.run)

if args.mode == "cv":
    is_CROSS_VALIDATION = True
elif args.mode == "out":
    is_CROSS_VALIDATION = False
else:
    print("invalid mode")
    os._exit()

if args.n2v != "none":
    node2vec_file = args.n2v
else:
    print("invalid n2v")
    os._exit()

crossp = "cross/"
modelp = "model/"
resultp = "result/"
fold = 5

def split_df(df, fold):
    cv = os.path.exists(crossp)
    if not cv:
        os.makedirs(crossp)
    for i in range(fold):
        path = crossp + str(i)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        fasta_train = open(path + "/train_fasta", "a")
        fasta_test = open(path + "/test_fasta", "a")
        label_train = open(path + "/train_label", "a")
        label_test = open(path + "/test_label", "a")
        for index, row in df.iterrows():
            if index % fold == i:
                fasta_test.write('>' + str(row["id"]) + '\n')
                fasta_test.write(str(row["seq"]) + '\n')
                label_test.write(str(row["target"]) + '\n')
            else:
                fasta_train.write('>' + str(row["id"]) + '\n')
                fasta_train.write(str(row["seq"]) + '\n')
                label_train.write(str(row["target"]) + '\n')
        print("- fold "+str(i)+" finish")
    print("----- finish fold dataset generation, start trainning -----")
    fasta_train.close()
    fasta_test.close()
    label_train.close()
    label_test.close()

def train_and_test(df):
    ps = os.path.exists("out_test/")
    if not ps:
        os.makedirs("out_test/")
    os.system('copy /y E:\Code\\Graph-Root\independent_testset\\test_label E:\Code\\Graph-Root\out_test')       # remember to change your path
    os.system('copy /y E:\Code\\Graph-Root\independent_testset\\test_fasta E:\Code\\Graph-Root\out_test')
    fasta_train = open(out_test + "/train_fasta", "a")
    label_train = open(out_test + "/train_label", "a")
    for index, row in df.iterrows():
        fasta_train.write('>' + str(row["id"]) + '\n')
        fasta_train.write(str(row["seq"]) + '\n')
        label_train.write(str(row["target"]) + '\n')
    fasta_train.close()
    label_train.close()
    res = os.path.exists("train_result.txt")
    if res:
        os.remove("train_result.txt")
    res = os.path.exists("predict_result.txt")
    if res:
        os.remove("predict_result.txt")
    result = os.path.exists(resultp)
    model = os.path.exists(modelp)
    if not model:
        os.makedirs(modelp)
    if not result:
        os.makedirs(resultp)
    train.main("114514", out_test, node2vec_file)
    predict.main("114514", out_test, node2vec_file)

def remove_dir(dir):
    dir = dir.replace('\\', '/')
    if(os.path.isdir(dir)):
        for p in os.listdir(dir):
            remove_dir(os.path.join(dir,p))
        if(os.path.exists(dir)):
            os.rmdir(dir)
    else:
        if(os.path.exists(dir)):
            os.remove(dir)

def start_train():
    res = os.path.exists("train_result.txt")
    if res:
        os.remove("train_result.txt")
    model = os.path.exists(modelp)
    if not model:
        os.makedirs(modelp)
    for i in range(fold):
        datapath = crossp + str(i)
        mdname = str(i)
        train.main(mdname, datapath, node2vec_file)
        print("----- train finish : "+str(i)+" -----")
    print("----- all train finish, start perdict -----")

def start_predict():
    res = os.path.exists("predict_result.txt")
    if res:
        os.remove("predict_result.txt")
    result = os.path.exists(resultp)
    if not result:
        os.makedirs(resultp)
    for i in range(fold):
        datapath = crossp + str(i)
        mdname = str(i)
        predict.main(mdname, datapath, node2vec_file)
        print("----- predict finish : "+str(i)+" -----")
    print("----- all predict finish, start perdict -----")

def cv_result():
    res = pd.read_table("predict_result.txt",names=["MCC", "F-score", "Precision", "Recall", "Accuracy", "sensitivity", "specificity", "AUC"])
    mean = res.mean().tolist()
    print("result of cross validation : \n")
    print(mean)
    with open("final_result.txt", "a") as f:
        for i in range(len(mean)):
            if i == len(mean)-1:
                f.write(str(mean[i]) + "\n")
            else:
                f.write(str(mean[i]) + "\t")

def main():
    DF = pd.concat([dfpositive, dfnegative.sample(len(dfpositive), random_state=random.randint(1000,99999))], ignore_index=True)
    if is_CROSS_VALIDATION:
        remove_dir(crossp)
        remove_dir(modelp)
        remove_dir(resultp)
        print("===== random negative set get, generating "+str(fold)+" fold dataset =====")
        split_df(DF, fold)
        start_train()
        start_predict()
        cv_result()

    else:
        print("===== random negative set get, generating train dataset, start out test =====")
        remove_dir(out_test)
        train_and_test(DF)
        cv_result()
    print("\n\n===== FINISH PROGRAM =====\n")


if __name__ == "__main__":
    for i in range(runtimes):
        main()
