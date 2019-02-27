from __future__ import print_function
from tqdm import tqdm
# from tqdm import tqdm_gui
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from model import word2vec
from utils_modified import load_data

CONTEXT_SIZE = 5
EMBEDDING_DIM = 30
NUM_EPOCHS = 15 #CHANGE

filename = "medium_text.txt"
# filename = 'tsts.txt'
print("Parsing text and loading training data...")
vocab, word_to_ix, ix_to_word, training_data = load_data(filename, CONTEXT_SIZE, model_type="skipgram", subsampling=True, sampling_rate=0.001)
print('len(training_data): ', len(training_data), '\n')

losses = []
loss_function = nn.NLLLoss()

# model = SkipGram(len(vocab), EMBEDDING_DIM)
model = word2vec(len(word_to_ix), EMBEDDING_DIM)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr = 0.008, momentum=0.9)
# print(model, '\n')
# print(optimizer, '\n')
# exit()
batch_size = 500
print("Starting training")
for epoch in tqdm(range(NUM_EPOCHS)):
    # total_loss = torch.Tensor([0])
    print("Beginning epoch %d/%d" % (epoch, NUM_EPOCHS))

    # '''
    training_data = np.array(training_data)
    #print(type(training_data), training_data.shape)
    # print(training_data.shape)
    num_batches = training_data.shape[0]/batch_size
    # print('num_batches: ', num_batches, '\n')
    # print(100/22)


    for batch in (range(num_batches + 1)):

        if batch < num_batches:

            # print(batch, batch*batch_size, batch*batch_size + batch_size)
            x_batch = training_data[batch*batch_size: batch*batch_size + batch_size , 0]
            x_batch = torch.tensor(x_batch).view(-1, 1)
            # print(x_batch.shape)
            # print(batch*batch_size + batch_size, x_batch.shape)
            target = torch.tensor(training_data[batch*batch_size: batch*batch_size + batch_size , 0])
            # print(target.shape)        
        else:
            # print(batch, batch*batch_size, training_data.shape[0])
            x_batch = training_data[batch*batch_size: training_data.shape[0] , 0]
            x_batch = torch.tensor(x_batch).view(-1, 1)
            # print(x_batch.shape)
            target = torch.tensor(training_data[batch*batch_size: batch*batch_size + batch_size , 0])   
            # print(target.shape)
    
        optimizer.zero_grad()
        log_probs = model(x_batch)
        # print('log_probs.shape: ', log_probs.shape)

        loss = loss_function(log_probs, target)

        loss.backward()
        optimizer.step()    
        # pdb.set_trace()

        losses.append(loss)
        # print('======')
        # exit()
    print("Epoch %d Loss: %.5f" % (epoch, loss))

print('\n')

EMBEDDINGS = model.embeddings.weight.data
print('EMBEDDINGS.shape: ', EMBEDDINGS.shape)

from sklearn.manifold import TSNE

print('\n', 'running TSNE...')
tsne = TSNE(n_components = 2).fit_transform(EMBEDDINGS)
print('tsne.shape: ', tsne.shape) #(15, 2)

############ VISUALIZING ############
x, y = [], []
annotations = []
for idx, coord in enumerate(tsne):
    # print(coord)
    annotations.append(ix_to_word[idx])
    x.append(coord[0])
    y.append(coord[1])   

# test_words = ['king', 'queen', 'berlin', 'capital', 'germany', 'palace', 'stays']
test_words = ['sun', 'moon', 'earth', 'while', 'open', 'run', 'distance', 'energy', 'coal', 'exploit']

plt.figure(figsize = (5, 5))
for i in range(len(test_words)):
    word = test_words[i]
    #print('word: ', word)
    vocab_idx = word_to_ix[word]
    # print('vocab_idx: ', vocab_idx)
    plt.scatter(x[vocab_idx], y[vocab_idx])
    plt.annotate(word, xy = (x[vocab_idx], y[vocab_idx]), \
        ha='right',va='bottom')

plt.savefig("w2v.png")
plt.show()

exit()

