from __future__ import print_function
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
import pdb
import numpy as np
# txt_file_path = '../input/medium_text.txt'
#txt_file_path = 'test.txt'
lem = WordNetLemmatizer()

def gather_word_freqs(split_text, subsampling = False, sampling_rate = 0.001): #here split_text is sent_list
    #print('\n', 'into gather_word_freqs...')
    #print('subsampling: ', subsampling)
    #print('num of sentences: ', len(split_text))
    vocab = {}
    ix_to_word = {}
    word_to_ix = {}
    total = 0.0

    for word_tokens in split_text:
        #print('SENTENCE: ', sent, '\n')
        #print('word_tokens: ', word_tokens)
        #print('========')
        for word in word_tokens: #for every word in the word list(split_text), which might occur multiple times
            if word not in vocab: #only new words allowed
                vocab[word] = 0
                ix_to_word[len(word_to_ix)] = word
                word_to_ix[word] = len(word_to_ix)
            vocab[word] += 1.0 #count of the word stored in a dict
            total += 1.0 #total number of words in the word_list(split_text)
    
    '''
    if subsampling:
        print('\n', 'subsampling is True')
        for sent in split_text:
            print('sent: ', sent)
            word_tokens = word_tokenize(sent)
            print(word_tokens)
            print('len(word_tokens): ', len(word_tokens))
            for i , word in enumerate(word_tokens):
                print(i, word_tokens[i])
                
                val = np.sqrt(sampling_rate * total / vocab[word])
                #print(vocab[word], val)
                #exit()
                # print('total, sampling_rate * total, vocab[word], val: ', total, sampling_rate * total, vocab[word], val)

                prob = val * (1 + val)
                sampling = np.random.sample()
                #print(sampling, prob)
                if (sampling <= prob):
                    print('freq: ', vocab[word_tokens[i]])
                    del word_tokens[i]
                    i -= 1
                print(i, word_tokens[i], '========')
                #print('========')
            #print(word_tokens)
            exit()
    '''

    return split_text, vocab, word_to_ix, ix_to_word

def gather_training_data(split_text, word_to_ix, context_size, model_type = "skipgram"):
    
    #print('\n', 'in gather_training_data...')
    #print('len(split_text): ', len(split_text)) #3238
    #print('context_size: ', context_size) #2

    training_data = []
    #for each sentence
    for sentence in split_text:
        #print('sentence: ', sentence)
        indices = [word_to_ix[word] for word in sentence]
        #print('indices: ', indices)
        # print('=========')
        #exit()
        #for each word treated as center word
        for center_word_pos in range(len(indices)):
            #print('center_word_pos: ', center_word_pos)
            #for each window  position
            for w in range(-context_size, context_size+1):
                #print('w: ', w)
                context_word_pos = center_word_pos + w
                #print('center_word_pos and context_word_pos: ', center_word_pos,  context_word_pos)
                #make sure we dont jump out of the sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                center_word_idx  = indices[center_word_pos]
                #print(context_word_idx)
                training_data.append([center_word_idx, context_word_idx])
                #print(center_word_idx, context_word_idx)
                
                ##print('------')
            #exit()
            #print('==================')
        #print('**************')
        #exit()

    return training_data

def load_data(filename, context_size, model_type = "skipgram", subsampling = False, sampling_rate = 0.001):
    print('into utils_modified !!!')
    with open(filename, "rb") as file:
        
        text = file.read().lower().decode('utf-8').strip()
        stop_words = set(stopwords.words('english')) 
        text_list = text.split(' ')
        # LEMMATIZING AND STOP WORDS
        text_list_without_stopwords = [lem.lemmatize(text_list[i], 'v') for i in range(len(text_list)) if text_list[i] not in stop_words]
        text_without_stopwords = ' '.join(text_list_without_stopwords)
        #print('text_without_stopwords: ', text_without_stopwords, '\n')
        
        sent_list = sent_tokenize(text_without_stopwords)
        #for s in sent_list:
            #print(s, '\n')
        
        sent_list_tokenized = [word_tokenize(s) for s in sent_list]
        #for s in sent_list_tokenized:
            #print(s, '\n')
              
        sent_list_tokenized, vocab, word_to_ix, ix_to_word = gather_word_freqs(sent_list_tokenized, subsampling = subsampling, sampling_rate = sampling_rate)        
        
        #print('len(word_to_ix): ', len(word_to_ix))
        #print('len(vocab): ', len(vocab))
        #print(ix_to_word, '\n')
        training_data = gather_training_data(sent_list_tokenized, word_to_ix, context_size,
                                             model_type = model_type)

        return vocab, word_to_ix, ix_to_word, training_data        
 

#load_data(txt_file_path, context_size = 2, subsampling = True)