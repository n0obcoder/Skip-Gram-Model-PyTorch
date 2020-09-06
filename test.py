import torch, sys, pdb, os
import numpy as np
from model import Word2Vec_neg_sampling
from utils_modified import nearest_word
from config import EMBEDDING_DIM, MODEL_DIR, DEVICE

def q():
    sys.exit()

def print_nearest_words(model, test_words, word_to_ix, ix_to_word, top = 5):
    
    model.eval()
    emb_matrix = model.embeddings_input.weight.data.cpu()
    
    nearest_words_dict = {}

    print('==============================================')
    for t_w in test_words:
        
        inp_emb = emb_matrix[word_to_ix[t_w], :]  

        emb_ranking_top, _ = nearest_word(inp_emb, emb_matrix, top = top+1)
        print(t_w.ljust(10), ' | ', ', '.join([ix_to_word[i] for i in emb_ranking_top[1:]]))

    return nearest_words_dict


if __name__ == '__main__':
    ckpt = torch.load(os.path.join(MODEL_DIR, 'model0.pth'))
    ix_to_word = ckpt['ix_to_word']
    word_to_ix = ckpt['word_to_ix']

    model = Word2Vec_neg_sampling(EMBEDDING_DIM, len(ix_to_word), DEVICE).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    EMBEDDINGS = model.embeddings_input.weight.data.cpu()
    print('EMBEDDINGS.shape: ', EMBEDDINGS.shape)

    def vec( word):
        return EMBEDDINGS[word_to_ix[word], :]

    inp = vec('king') - vec('man') + vec('woman')                                       
    print('inp.shape: ', inp.shape)

    emb_ranking_top, euclidean_dis_top = nearest_word(inp, EMBEDDINGS, top = 6)
    print('emb_ranking_top: ', emb_ranking_top, type(emb_ranking_top))

    for idx, t in enumerate(emb_ranking_top):
        print(ix_to_word[t], euclidean_dis_top[idx])
