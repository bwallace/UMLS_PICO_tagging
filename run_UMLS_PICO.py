'''
Run the fancy model.
'''

import pickle
#import unicodecsv as csv
import csv
import itertools
#import unidecode

import numpy as np 
import cPickle as pickle
import networkx as nx

import gensim
from gensim.models import Word2Vec

import UMLS_PICO

import keras


# pre-trained embeddings!
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

def main():
    # load the data
    with open('cui_data_sent.csv', 'r') as f:
        r = csv.DictReader(f)
        data = [row for row in r]
        
    # and the CUI graph
    with open('graph_subset.pck', 'rb') as f:
        cui_graph = pickle.load(f)


    # quick processing of cuis
    cui2int = lambda CUI_str: int(CUI_str[1:])
    int2cui = lambda CUI_int: "C{}".format(str(CUI_int).zfill(7))
    cui_ancestors = lambda CUI_str: list(nx.ancestors(cui_graph, cui2int(CUI_str)))

    '''
    # sanity check / test
    print "CUI: {cui}\t LABEL: {label}\nSENT: {sent}".format(**data[0])
    print "ANCESTORS: {}".format(','.join([int2cui(c) for c in cui_ancestors(data[0]['cui'])]))
    print data[0].keys()
    '''

    ancestors, texts, y = [], [], []
    #targets = []
    for row in data:
        cur_ancestors = [cui2int(row['cui'])] # the index CUI
        try:
            cur_ancestors = cur_ancestors + list(cui_ancestors(row['cui']))
        except:
            pass 
        
        ancestors.append(" ".join([int2cui(a) for a in cur_ancestors]))
        texts.append(row['sent'])
        y.append(row['label'])


    
    # initializiations
    CUI_vectors = load_CUI_vectors()
    word_vectors = load_token_vectors()

    # set up the preprocessor
    p = UMLS_PICO.Preprocessor(wvs=word_vectors, CUI_vs=CUI_vectors)
    p.preprocess(texts, ancestors)

    # instantiate the model
    m = UMLS_PICO.CUI_PICO_Mapper(p)

    # build features
    X_text = p.build_text_sequences(texts)
    X_CUI  = p.build_CUI_sequences(ancestors)

    ### temp!!!
    lbl_f = lambda y_i : [0, 1] if y_i == "interventions" else [1, 0]
    y_tmp = [lbl_f(y_i) for y_i in y]

    history = m.model.fit({'word_input':X_text, 'CUI_input':X_CUI, 'output':y_tmp})
