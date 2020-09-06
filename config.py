import os, torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('DEVICE: ', DEVICE)

DATA_SOURCE           = 'gensim' # or 'toy'
MODEL_ID              = DATA_SOURCE #'toy'# 'gensim'
DISPLAY_BATCH_LOSS    = True

if DATA_SOURCE=='toy':
    DISPLAY_EVERY_N_BATCH = 5000
    SAVE_EVERY_N_EPOCH    = 100
    BATCH_SIZE            = 32
    NUM_EPOCHS            = int(1e+3)

    CONTEXT_SIZE          = 3
    FRACTION_DATA         = 1
    SUBSAMPLING           = False
    SAMPLING_RATE         = 0.001
    NEGATIVE_SAMPLES      = 0 # set it to 0 if you don't want to use negative samplings  

    EMBEDDING_DIM         = 3
    LR                    = 0.001

    TEST_WORDS            = ['word1', 'word3', 'word6', 'word13', 'word14']
    TEST_WORDS_VIZ        = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10', 'word11', 'word12', 'word13', 'word14', 'word15']

elif DATA_SOURCE=='gensim':
    DISPLAY_EVERY_N_BATCH = 1000
    SAVE_EVERY_N_EPOCH    = 1
    BATCH_SIZE            = 1024*16
    NUM_EPOCHS            = 10

    CONTEXT_SIZE          = 5
    FRACTION_DATA         = 1
    SUBSAMPLING           = True
    SAMPLING_RATE         = 0.001
    NEGATIVE_SAMPLES      = 20 # set it to 0 if you don't want to use negative samplings  

    EMBEDDING_DIM         = 100
    LR                    = 0.001

    if FRACTION_DATA == 1:
        TEST_WORDS            = ['india', 'computer', 'gold', 'football', 'cars', 'war', 'apple', 'music', 'helicopter']
        TEST_WORDS_VIZ        = ['india', 'asia', 'guitar', 'piano', 'album', 'music', 'war', 'soldiers', 'helicopter']
    else:
        TEST_WORDS            = ['human', 'boy', 'office', 'woman']
        TEST_WORDS_VIZ        = TEST_WORDS

PREPROCESSED_DATA_DIR  = os.path.join(MODEL_ID, 'preprocessed_data')
PREPROCESSED_DATA_PATH = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_data_' + MODEL_ID + '_' + str(FRACTION_DATA) + '.pickle')
SUMMARY_DIR            = os.path.join(MODEL_ID, 'summary') 
MODEL_DIR              = os.path.join(MODEL_ID, 'models')
