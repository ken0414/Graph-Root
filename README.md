# Graph-Root
Root-Associated Proteins in Plants Prediction Model Based on Graph Convolutional Network
## Introduction
In this repository, a model for prediction of root-related proteins in plant is provided. In this model, each protein is represented by features derived from its sequence and a protein network. The sequence features are refined from the raw features of amino acids, which are processed by graph convolutional network and multi-head attention, whereas the network features are extracted from a protein network via Node2vec. All these features are fed into the fully connected layer to make prediction.
![Image text](https://github.com/ken0414/Graph-Root/blob/89dea32b58ffdbcb184772ed553abdbc61cf010f/image/Figure.jpg)
## Requirements
To run this program, you may need:
 * Python 3.6 or later
 * Pytorch 1.12.1 and other related packages
 * Windows 10 enviroment
 * GPU (optional for cuda)
## How to use
1. Set up your enviroment and download the code from github:
  ```
     git clone https://github.com/ken0414/Graph-Root.git
  ```
2. Put your data into the appropriate folder:
  ```
     [network embedding feature(node2vec)] --> ./data/n2v
     [domain feature(IPR)] --> ./data/ipr
     [protein structure feature] --> ./data/graph    
     [protein node feature(PSSM)] --> ./data/pssm
  ```
3. Activate your enviroment and run main.py:
  ```
     $ python main --mode cv --n2v example_n2v --run 10
  ```
  Within this line of code, you can choose between cross-validation mode or independent test set mode by modifying the value after mode, n2v to change the file used, and run to select the number of repetitive runs.
| option  | value |
| ------------- | ------------- |
| `mode` | `cv`or`out` |
| `n2v` | filename of node2vec |
| `run` | int value for run times |
4. Get result:
  For each fold in a single cv, you can get the best epoch of the train in `train_result.txt`.
  After all fold trained in a single cv, you can get the evaluation of all fold in `predict_result.txt` and the result of prediction in fold `./result`.
  If you run on `out` mode, there will be only 1 result in `train_result.txt` and `predict_result.txt`.
  If the value of `run` is bigger than 1, former result in `train_result.txt` of a single `cv` or `out` will be override. If you want to save the result of this file, please modify the code on your own.
  Samely, if you run the main.py again, all the result will be override.
