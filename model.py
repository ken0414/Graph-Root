from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import math

# GCN parameters
GCN_FEATURE_DIM = 40
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 64
ATTENTION_HEADS = 64

# Node2vec parameters
NODE2VEC_FEATURE_DIM = 512

# Final linear layer parameters
FINAL_LINEAR_DIM = 2048

# Training parameters
LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1E-4

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):      
        support = input @ self.weight         
        output = adj @ support           
        if self.bias is not None:        
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(GCN_FEATURE_DIM, GCN_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(GCN_HIDDEN_DIM)
        self.gc2 = GraphConvolution(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.ln2 = nn.LayerNorm(GCN_OUTPUT_DIM)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, adj):  			
        x = self.gc1(x, adj)  				
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))	
        return output


class Attention(nn.Module):

    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				
        x = torch.tanh(self.fc1(input))  	
        x = self.fc2(x)  					
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		
        return attention

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.gcn = GCN()
        self.attention = Attention(GCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_1 = nn.Linear(GCN_OUTPUT_DIM * ATTENTION_HEADS + NODE2VEC_FEATURE_DIM, FINAL_LINEAR_DIM)
        self.fc_2 = nn.Linear(FINAL_LINEAR_DIM, 1)  
        self.fc_n = nn.Linear(GCN_OUTPUT_DIM * ATTENTION_HEADS + NODE2VEC_FEATURE_DIM, 1)                  
        self.criterion = nn.BCELoss()                                                  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, adj, n2v):  											
        
        x = x.float()                                                                   # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        adj = adj.float()
        x = self.gcn(x, adj)  												            # x.shape = (seq_len, GAT_OUTPUT_DIM)

        x = x.unsqueeze(0).float()  										            # x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)  											            # att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x 									            # output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        
        n2v = n2v.unsqueeze(0).float()
        node_feature_embedding_con = torch.flatten(node_feature_embedding, start_dim=1) # output.shape = (1, ATTENTION_HEADS * GAT_OUTPUT_DIM)
        embedding_and_n2v_feature = torch.cat((node_feature_embedding_con, n2v), dim=1) # output.shape = (1, ATTENTION_HEADS * GAT_OUTPUT_DIM + NODE2VEC_FEATURE_DIM)
        fc1_feature = self.fc_1(embedding_and_n2v_feature)                              # output.shape = (1, FINAL_LINEAR_DIM)
        output = torch.sigmoid(self.fc_2(fc1_feature)).squeeze(0)  	                    # output.shape = (1)                

        return output