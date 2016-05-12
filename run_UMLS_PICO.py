'''
Run the fancy model.
'''
import random
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
    random.seed(1337)
    
    # load the data
    with open('cui_data_sent5.csv', 'r') as f:
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
        

        ancestors_list = " ".join([int2cui(a) for a in cur_ancestors])
        #ancestors.append(" ".join([int2cui(a) for a in cur_ancestors]))
        #texts.append(row['sent'])
        #y.append(row['label'])


        #import pdb; pdb.set_trace()
        # because each sentence will have multiple
        # labels (separated by `|' chars), we need
        # to split these
        for label in row['label'].split('|'):
            #new_row = row.copy()
            #new_row['label'] = label
            #ancestors.append(" ".join([int2cui(a) for a in cur_ancestors]))
            texts.append(row['sent'])
            ancestors.append(ancestors_list)
            y.append(label)

    
    import pdb; pdb.set_trace()
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
   
    '''

    @TODO you need to merge all of these and deal witht the ORs (|)!!!! (see below)
    (Pdb) set(y_train)
    set([u'primary_outcome|secondary_outcome', u'interventions', u'interventions|primary_outcome', u'population|interventions|primary_outcome', u'interventions|primary_outcome|secondary_outcome', u'interventions|secondary_outcome', u'ignore', u'population|primary_outcome|secondary_outcome', u'population|interventions', u'population|interventions|primary_outcome|secondary_outcome', u'population|primary_outcome', u'population|secondary_outcome', u'outcome', u'population|interventions|secondary_outcome', u'population'])

    [u'primary_outcome|secondary_outcome',
     u'interventions',
     u'interventions|primary_outcome',
     u'population|interventions|primary_outcome',
     u'interventions|primary_outcome|secondary_outcome',
     u'interventions|secondary_outcome',
     u'ignore',
     u'population|primary_outcome|secondary_outcome',
     u'population|interventions',
     u'population|interventions|primary_outcome|secondary_outcome',
     u'population|primary_outcome',
     u'population|secondary_outcome',
     u'outcome',
     u'population|interventions|secondary_outcome',
     u'population']

     @ iain says
     out = []
    for row in data:    
        for label in row['label'].split('|'):
            new_row = row.copy()
            new_row['label'] = label
            out.append(new_row)
    ''' 
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

    # calculate class weights (these are to be inverse to prevalence, as estimated in training data!)
    
    '''
    y_train = y_mat[train,:]
    
    class_counts = np.sum(y_train, axis=0)
    N = float(len(train))
    prevalences = class_counts/N
    weights = 1.0/prevalences
    weights_d = dict(zip(range(4), weights.tolist()[0]))
    '''

    ''' 4/5 -- experimental '''
    # sampling -- experimental!
    class_instance_indices = {}
    for lbl, j in lbl_vectorizer.vocabulary_.items():
        #indices = [idx for idx in range(y_train.shape[0]) if y_train[idx,j]>0]
        indices = [idx for idx in train if y_mat[idx,j]>0]
        class_instance_indices[lbl] = indices
    
    # 10x as many 'ignores' as smallest class; this is arbitrary
    num_population = len(class_instance_indices["population"])
    ignore_indices = random.sample(class_instance_indices["ignore"], 10*num_population)
    outcome_indices = class_instance_indices["outcome"]
    intervention_indices = class_instance_indices["interventions"]
    population_indices = class_instance_indices["population"]

    # now sample the rest evenly
    #outcome_indices = random.sample(class_instance_indices["outcome"], num_population)
    #intervention_indices = random.sample(class_instance_indices["interventions"], num_population)
    #population_indices = class_instance_indices["population"]


    downsampled_train_indices = ignore_indices + outcome_indices + intervention_indices + population_indices
    random.shuffle(downsampled_train_indices)
    train = downsampled_train_indices

    y_train = y_mat[train,:]
    
    class_counts = np.sum(y_train, axis=0)
    N = float(len(train))
    prevalences = class_counts/N
    weights = 1.0/prevalences
    weights_d = dict(zip(range(4), weights.tolist()[0]))

    # dump 
    with open("fold_%s_train_ids.pickle" % fold_num, 'w') as output_f:
        pickle.dump(train, output_f)


    
    #y_train = y_mat[train,:]
    #y_train = y_mat[downsampled_train_indices,:]

    #import pdb; pdb.set_trace()
    ###
    # 4/4/2016 -- consider down-sampling here?
    #import pdb; pdb.set_trace()



    with open("fold_%s_test_ids.pickle" % fold_num, 'w') as output_f:
        pickle.dump(test, output_f)

    print "using class weights!"
    history = m.model.fit({'word_input':X_text[train], 
                            'CUI_input':X_CUI[train], 
                            'output':y_mat[train,:]}, 
                            nb_epoch=50, class_weight=weights_d)
                            #'output':y_tmp[train]}, nb_epoch=1)

    predictions = m.model.predict({'word_input':X_text[test], 
                                    'CUI_input':X_CUI[test]},
                                    batch_size=128)

    json_string = m.model.to_json()
    open('model_architecture.json', 'w').write(json_string)

    print "dumping weights!"
    m.model.save_weights('model_weights_redux_%s.h5' % fold_num, overwrite=True)
    print "done!"

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
