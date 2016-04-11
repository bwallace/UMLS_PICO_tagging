from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
from scipy.sparse import csr_matrix, hstack
import unicodecsv as csv
import cPickle as pickle
import networkx as nx
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import fbeta_score, make_scorer
import sys 

import pdb
import unidecode

def main(fold_num=0):
    train_ids = pickle.load(open("fold_%s_train_ids.pickle" % fold_num))
    test_ids  = pickle.load(open("fold_%s_test_ids.pickle" % fold_num))

    # get the data
    with open('cui_data_sent.csv', 'r') as f:
        r = csv.DictReader(f)
        data = [row for row in r]
        
    # cui graph
    with open('graph_subset.pck', 'rb') as f:
        cui_graph = pickle.load(f)

    # de-unicode all this
    
    for row in data:
        row["sent"] = unidecode.unidecode(row["sent"])

    # quick processing of cuis
    cui2int = lambda x: int(x[1:])
    int2cui = lambda x: "C{}".format(str(x).zfill(7))
    cui_ancestors = lambda x: list(nx.ancestors(cui_graph, cui2int(x)))

    # Generate text features
    vec = HashingVectorizer(ngram_range=(1,3), stop_words='english')
    X_text = vec.transform((row['sent'] for row in data))

    # Generate concept features
    indptr = [0]
    indices = []
    csr_data = []

    for row in data:
        
        ancestors = [cui2int(row['cui'])]# remember the index cui!
        try:
            ancestors = ancestors + cui_ancestors(row['cui'])
        except:
            pass
            
        for ancestor in ancestors:
            indices.append(ancestor)
            csr_data.append(1)
        indptr.append(len(indices))

    X_cuis = csr_matrix((csr_data, indices, indptr), shape=(len(data), 10000000), dtype=np.int64)

    # and positional features
    X_pos = np.zeros(shape=(len(data), 5))
    for i, row in enumerate(data):
        X_pos[(i, int(float(row['position'])*5))] = 5

    # and answers
    y = np.array([row["label"] for row in data])

    # combine primary and secondary outcome for now (not sure it matters too much at this stage)
    y[y=='secondary_outcome'] = 'outcome'
    y[y=='primary_outcome'] = 'outcome'

    X = hstack([X_text, X_cuis, X_pos], format='csr')


    X_train = X[train_ids,:]
    X_test  = X[test_ids,:]

    y_train = y[train_ids]
    y_test  = y[test_ids]

    import pdb; pdb.set_trace()
    class_instance_indices = {}
    outcome_indices = np.where(y_train == "outcome")[0]
    interventions_indices = np.where(y_train == "interventions")[0]
    ignore_indices = np.where(y_train == "ignore")[0]
    population_indices = np.where(y_train == "population")[0]

    K=5
    targets = ['ignore', 'population', 'interventions', 'outcome']

    #ftwo_scorer = make_scorer(fbeta_score, beta=2, labels=targets, average='macro') # favour recall a 
    # bcw -- making comparable to CNN approach
    f_scorer = make_scorer(f_scorer, beta=1, labels=targets, average='macro')

    
    '''
    class_weights = []
    # generate hyperparameter search space
    weight_space = range(1, 50) 

    for w1 in weight_space:
        for w2 in weight_space:
            for w3 in weight_space:
                class_weights.append({t: w for t, w in zip(targets, [w1, w2, w3])})
    '''

    parameters = {'alpha': np.logspace(-1, -20, 50)}

    clf = SGDClassifier(average=True, loss="hinge", class_weights="balanced")

    # do the random grid search thing
    grid_search = RandomizedSearchCV(clf, param_distributions=parameters, n_iter=25, 
                                        verbose=3, scoring=f_scorer, cv=K)

    grid_search.fit(X_train, y_train)

    #y_hat = grid_search.predict(X_test)
    y_hat = grid_search.decision_function(X_test)
    with open("lm_predictions_%s.pickle" % fold_num) as outf:
        pickle.dump(y_hat, outf)

    with open("lm_y_%s.pickle" % fold_num) as outf:
        pickle.dump(y_test, outf)


if __name__ == '__main__':
    #
    if len(sys.argv) > 1:
        fold = int(sys.argv[1])
        print "LINEAR MODEL running fold: %s" % fold 

    main(fold)



