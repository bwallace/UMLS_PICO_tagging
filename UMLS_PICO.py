import numpy as np 
import pandas as pd 

import gensim
from gensim.models import Word2Vec

import keras
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import accuracy
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint

import theano
theano.config.DebugMode = True

def load_CUI_vectors():
    ''' 
    From De Vine et al., CIKM 2014
    https://github.com/clinicalml/embeddings
    '''
    m = Word2Vec.load_word2vec_format("DeVine_etal_200.txt.gz")
    return m 

def load_token_vectors(path="/Users/byron/dev/Deep-PICO/PubMed-w2v.bin"):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m 


class Preprocessor:
    # @TODO setting max_CUI_size to something small for now!
    def __init__(self, max_vocab_size=10000, max_CUI_size=5000, 
                    max_len=40, max_CUI_len=100, 
                    wv_embedding_dims=200, CUI_embedding_dims=200,
                    wvs=None, CUI_vs=None):
        '''
        max_vocab_size: maximum number of words to include in the model
        max_CUI_size: maximum number of CUIs to include in the model
        max_features: the upper bound to be placed on the vocabulary size.
        max_len: the maximum length (in terms of tokens) of the text snippets.
        max_CUI_size: the maximum number of ancestral CUIs to be used for each instance.
        wv_embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors are provided (if wvs is not None).
        CUI_embedding_dims: size of the CUI embeddings; over-ridden if pre-trained
                          vectors are provided. 
        wvs: pre-trained embeddings (for embeddings initialization)
        '''

        # inputs
        self.max_vocab_size = max_vocab_size
        self.max_CUI_size   = max_CUI_size

        self.max_len = max_len 
        self.max_CUI_len = max_CUI_len 

        self.tokenizer = Tokenizer(nb_words=self.max_vocab_size)
        # overkill to use a tokenizer, but we'll do it anyway
        self.CUI_tokenizer = Tokenizer(nb_words=self.max_CUI_size)

        self.use_pretrained_embeddings = False
        self.init_vectors = None 
        if wvs is None:
            self.wv_embedding_dims = wv_embedding_dims
            self.CUI_embedding_dims = CUI_embedding_dims
        else:
            # note that these are only for initialization;
            # they will be tuned!
            self.use_pretrained_embeddings = True

            self.word_embeddings = wvs
            self.wv_embedding_dims = wvs.vector_size

            self.CUI_embeddings = CUI_vs
            self.CUI_embedding_dims = CUI_vs.vector_size


    def preprocess(self, all_texts, all_CUIs):
        ''' 
        This fits tokenizer and builds up input vectors (X) from the list 
        of texts in all_texts. Needs to be called before train!
        '''
        self.raw_texts = all_texts
        self.CUIs = all_CUIs
        self.fit_tokenizer()
        self.fit_CUI_tokenizer()
        if self.use_pretrained_embeddings:
            print "initializing word vectors.."
            self.init_word_vectors()
            print "done. initializing CUI vectors..."
            self.init_CUI_vectors()
            print "done."


    def fit_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

    def fit_CUI_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.CUI_tokenizer.fit_on_texts(self.CUIs)
        self.CUI_indices_to_CUIs = {}
        for CUI, idx in self.CUI_tokenizer.word_index.items():
            self.CUI_indices_to_CUIs[idx] = CUI

    def build_text_sequences(self, texts):
        X = list(self.tokenizer.texts_to_sequences_generator(texts))
        X = np.array(pad_sequences(X, maxlen=self.max_len))
        return X

    def build_CUI_sequences(self, CUIs):
        X_CUIs = list(self.CUI_tokenizer.texts_to_sequences_generator(CUIs))
        X_CUIs = np.array(pad_sequences(X_CUIs, maxlen=self.max_CUI_len))
        return X_CUIs

    def init_word_vectors(self):
        ''' 
        Initialize word vectors.
        '''
        self.init_word_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_vocab_size:
                try:
                    self.init_word_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.wv_embedding_dims)*-2 + 1

                    self.init_word_vectors.append(unknown_words_to_vecs[t])

        # note that we make this a singleton list because that's
        # what Keras wants. 
        self.init_word_vectors = [np.vstack(self.init_word_vectors)]


    def init_CUI_vectors(self): 
        '''
        initialize CUI vectors 
        '''
        self.init_CUI_vectors = []
        unknown_CUIs_to_vecs = {}
        for CUI, CUI_idx in self.CUI_tokenizer.word_index.items():
            if CUI_idx <= self.max_CUI_size:
                try: 
                    self.init_CUI_vectors.append(self.CUI_embeddings[CUI])
                except:
                    if CUI not in unknown_CUIs_to_vecs:
                        unknown_CUIs_to_vecs[CUI] = np.random.random(self.CUI_embedding_dims)*-2 + 1
                    self.init_CUI_vectors.append(unknown_CUIs_to_vecs[CUI])

        self.init_CUI_vectors = [np.vstack(self.init_CUI_vectors)]


class CUI_PICO_Mapper:

    def __init__(self, preprocessor, filters=None, n_filters=50, dropout=0.1):
        self.preprocessor = preprocessor

        if filters is None:
            self.ngram_filters = [2, 3, 5]
        else:
            self.ngram_filters = filters 

        self.nb_filter = n_filters 
        self.dropout = dropout

        self.build_model() 

    def build_model(self):
        self.model = Graph()


        ############################
        # Word inputs/convolutions #
        ############################
        self.model.add_input(name='word_input', input_shape=(self.preprocessor.max_len,), dtype=int)

        # embedding layer for words
        self.model.add_node(Embedding(self.preprocessor.max_vocab_size, self.preprocessor.wv_embedding_dims, 
                                input_length=self.preprocessor.max_len, 
                                weights=self.preprocessor.init_vectors), 
                                name='word_embedding', input='word_input')

        for n_gram in self.ngram_filters:
            self.model.add_node(Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.preprocessor.wv_embedding_dims,
                                         input_length=self.preprocessor.max_len),
                           name='words_conv_' + str(n_gram),
                           input='word_embedding')

            self.model.add_node(MaxPooling1D(pool_length=self.preprocessor.max_len - n_gram + 1),
                           name='maxpool_words_' + str(n_gram),
                           input='words_conv_' + str(n_gram))

            self.model.add_node(Flatten(),
                           name='flat_words_' + str(n_gram),
                           input='maxpool_words_' + str(n_gram))
        
        self.model.add_node(Dropout(self.dropout), name='words_dropout', 
            inputs=['flat_words_' + str(n) for n in self.ngram_filters])
        
        self.model.add_node(Dense(1, input_dim=self.nb_filter * len(self.ngram_filters)), 
                                  name='words_dense', input='words_dropout')


        ###########################
        # CUI inputs/convolutions #
        ###########################
        self.model.add_input(name='CUI_input', input_shape=(self.preprocessor.max_CUI_len,), dtype=int)

        self.model.add_node(Embedding(self.preprocessor.max_CUI_size, self.preprocessor.CUI_embedding_dims, 
                                input_length=self.preprocessor.max_CUI_len, 
                                weights=self.preprocessor.init_CUI_vectors), 
                                name='CUI_embedding', input='CUI_input')

        
        for n_gram in self.ngram_filters:
            self.model.add_node(Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.preprocessor.CUI_embedding_dims,
                                         input_length=self.preprocessor.max_CUI_len),
                           name='CUI_conv_' + str(n_gram),
                           input='CUI_embedding')

            self.model.add_node(MaxPooling1D(pool_length=self.preprocessor.max_CUI_len - n_gram + 1),
                           name='maxpool_CUI_' + str(n_gram),
                           input='CUI_conv_' + str(n_gram))

            self.model.add_node(Flatten(),
                           name='flat_CUI_' + str(n_gram),
                           input='maxpool_CUI_' + str(n_gram))

        self.model.add_node(Dropout(self.dropout), name='CUIs_dropout', 
            inputs=['flat_CUI_' + str(n) for n in self.ngram_filters])

        self.model.add_node(Dense(1, input_dim=self.nb_filter * len(self.ngram_filters)), 
                                  name='CUIs_dense', input='CUIs_dropout')


        ##############################
        #  Merge CUI + word layers   #
        ##############################
        #self.model.add_node(Dense(1, input_dim=2*(self.nb_filter * len(self.ngram_filters)), 
        #                      name='final_dense', inputs=['words_dropout', 'CUIs_dropout']))
        

        self.model.add_node(Activation("sigmoid"), name='sigmoid', 
                                inputs=['words_dense', 'CUIs_dense'], 
                                merge_mode="concat")


        #self.model.add_node(Merge(['words_dropout', 'CUIs_dropout'], mode='concat')) #name='final_dense', 
        
        # and now the final classification
        
        # was (3/26)
        #self.model.add_output(name='output', input='sigmoid')
        self.model.add_node(Dense(4, activation='softmax'), 
                        name='output', input='sigmoid', 
                        create_output=True)


        print("model built")
        print(self.model.summary())
        #self.model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
        self.model.compile(optimizer='adam', loss={'output': 'categorical_crossentropy'})


