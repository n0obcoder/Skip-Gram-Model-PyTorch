import numpy as np
import sys

def q():
    sys.exit()

# define a function to count the total number of trainable parameters
def count_parameters(model): 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

# TEST
def nearest_word(inp, emb, top = 5, debug = False):
    euclidean_dis = np.linalg.norm(inp - emb, axis = 1)    
    emb_ranking = np.argsort(euclidean_dis)
    emb_ranking_distances = euclidean_dis[emb_ranking[:top]]
    
    emb_ranking_top = emb_ranking[:top]
    euclidean_dis_top = euclidean_dis[emb_ranking_top]
    
    if debug:
        print('euclidean_dis: ', euclidean_dis)
        print('emb_ranking: ', emb_ranking)
        print(f'top {top} embeddings are: {emb_ranking[:top]} with respective distances\n {euclidean_dis_top}')
    
    return emb_ranking_top, euclidean_dis_top
