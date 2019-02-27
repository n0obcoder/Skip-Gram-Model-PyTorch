from __future__ import print_function
import torch
#import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

class word2vec(nn.Module):
    def __init__(self, vocabulary_size, embedding_dims):
        super(word2vec, self).__init__()
        
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dims)
        # self.i2h = nn.Linear(vocabulary_size, embedding_dims)
        self.l1 = nn.Linear(embedding_dims, 32)
        self.l2 = nn.Linear(32, vocabulary_size)
        self.softmax = nn.LogSoftmax(dim=1)#(dim=1) #its not Softmax
        self.dropout = nn.Dropout(p= 0.1)
        
    def forward(self, input):

        # print('\n', 'input.shape: ', input.shape)
        
        embeddings = self.embeddings(input)
        # Input: LongTensor (N, W), N = mini-batch, W = number of indices to extract per mini-batch
        # Output: (N, W, embedding_dim)
        # print('embeddings.shape: ', embeddings.shape)
        # exit()
        embeddings = embeddings.squeeze()
        # print('embeddings.shape: ', embeddings.shape) #(70, 5)
        
        # exit()
        out1 = self.dropout(self.l1(F.relu(embeddings)))                                     #ReLU here too ???
        # print('out1.shape: ', out1.shape) #(70, 15)
        out2 = self.dropout(self.l2(F.relu(out1)))
        # print('out2.shape: ', out2.shape)
        log_probs = self.softmax(out2)
        # print('log_probs.shape: ', log_probs.shape) #(70, 15)
        # exit()
        return log_probs
    
    def predict(self, test_input):
        word_embedding = self.i2h(test_input)
        return word_embedding
