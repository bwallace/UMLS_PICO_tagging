'''
Run the fancy model.
'''
import pdb 
import sys
import pickle
#import unicodecsv as csv
import csv
import itertools
#import unidecode


import numpy as np 
import cPickle as pickle
import networkx as nx

import sklearn 
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

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

def main(fold_num):

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

    # and labels
    ### temp!!!
    #lbl_f = lambda y_i : [0, 1] if y_i == "interventions" else [1, 0]
    #y_tmp = np.array([lbl_f(y_i) for y_i in y])
    
    # merge 'primary' and 'secondary' outcomes
    lbl_f = lambda y_i : "outcome" if "outcome" in y_i else y_i
    y = [lbl_f(y_i) for y_i in y]

    # create a label vectorizer 
    lbl_vectorizer = CountVectorizer(binary=True, min_df=1)
    y_mat = lbl_vectorizer.fit_transform(y).todense()

    # dump the vectorizer
    with open("y_vectorizer_%s" % fold_num, 'w') as output_f:
        pickle.dump(lbl_vectorizer, output_f)

    # cross-fold validation; setup to parallelize
    # assume that we are only running fold n_fold
    # here
    n_folds = 5
    #import pdb; pdb.set_trace()
    #pdb.set_trace()
    skf = list(KFold(y_mat.shape[0], n_folds=n_folds, shuffle=True, random_state=1337))
    train, test = skf[fold_num]


    # dump
    with open("fold_%s_train_ids.pickle" % fold_num, 'w') as output_f:
        pickle.dump(train, output_f)

    with open("fold_%s_test_ids.pickle" % fold_num, 'w') as output_f:
        pickle.dump(test, output_f)

    history = m.model.fit({'word_input':X_text[train], 
                            'CUI_input':X_CUI[train], 
                            'output':y_mat[train,:]}, nb_epoch=1)
                            #'output':y_tmp[train]}, nb_epoch=1)

    predictions = m.model.predict({'word_input':X_text[test], 
                                    'CUI_input':X_CUI[test]},
                                    batch_size=64)


    with open("fold_%s_predictions.pickle" % fold_num, 'w')  as output_f:
        pickle.dump(predictions, output_f)

    with open("fold_%s_truth.pickle" % fold_num, 'w')  as output_f:
        pickle.dump(y_mat[test,:], output_f)

if __name__ == '__main__':
    #
    if len(sys.argv) > 1:
        fold = int(sys.argv[1])
        print "running fold: %s" % fold 


    main(fold)
