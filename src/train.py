import os
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import lightgbm as lgbm

def main(args):
    
    data = pd.read_csv(os.path.join(args.data_path,"output-vectorize.csv"))
    data['vectors'] = data['vectors'].apply(lambda row: [float(t) for t in row.split(';')])

    X_train = data.vectors.values
    X_train = [np.array(tmp) for tmp in X_train]
    X_train = np.array(X_train)
    
    y_train = data.sentiment.values

    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2, stratify=y_train, random_state = 42)

    # Get the train and test data for the training sequence
    train_data = lgbm.Dataset(X_train, label=y_train)
    test_data = lgbm.Dataset(X_test, label=y_test)

    '''
    # Parameters we'll use for the prediction
    parameters = {
        'application': 'binary',
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 0
    }

    classifier = lgbm.train(parameters,
                    train_data,
                    valid_sets= test_data,
                    num_boost_round=100,
                    early_stopping_rounds=10)
    '''

    fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgbm.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}

    param_test ={'num_leaves': sp_randint(6, 50), 
                'min_child_samples': sp_randint(100, 500), 
                'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                'subsample': sp_uniform(loc=0.2, scale=0.8), 
                'boosting_type':['gbdt','goss'],
                'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

    #This parameter defines the number of HP points to be tested
    n_HP_points_to_test = 100

    #n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
    clf = lgbm.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test, 
        n_iter=n_HP_points_to_test,
        scoring='roc_auc',
        cv=5,
        refit=True,
        random_state=314,
        verbose=True)

    gs.fit(X_train, y_train, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


    clf_sw = lgbm.LGBMClassifier(**clf.get_params())
    #set optimal parameters
    clf_sw.set_params(**gs.best_estimator_.get_params())
    gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                    param_grid={'scale_pos_weight':[1,2,6,12]},
                                    scoring='roc_auc',
                                    cv=5,
                                    refit=True,
                                    verbose=True)

    gs_sample_weight.fit(X_train, y_train, **fit_params)
    print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))



    #Configure from the HP optimisation
    clf_final = lgbm.LGBMClassifier(**clf_sw.get_params())
    res = clf_final.set_params(**gs_sample_weight.best_estimator_.get_params())
    print(res)
    def learning_rate_010_decay_power_099(current_iter):
        base_learning_rate = 0.1
        lr = base_learning_rate  * np.power(.99, current_iter)
        return lr if lr > 1e-3 else 1e-3

    def learning_rate_010_decay_power_0995(current_iter):
        print(current_iter)

        base_learning_rate = 0.1
        lr = base_learning_rate  * np.power(.995, current_iter)
        return lr if lr > 1e-3 else 1e-3

    def learning_rate_005_decay_power_099(current_iter):
        base_learning_rate = 0.05
        lr = base_learning_rate  * np.power(.99, current_iter)
        return lr if lr > 1e-3 else 1e-3

    #Train the final model with learning rate decay
    #clf_final.fit(X_train, y_train, **fit_params, 
    #			callbacks=[lgbm.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])
    clf_final.fit(X_train, y_train, **fit_params)

    return clf_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path'
    )    
    args = parser.parse_args()
    
    result = main(args)

    result.booster_.save_model(os.path.join(args.output, 'best-model.txt'))

    #print(result.head(5))
    #result.to_csv(os.path.join(args.output,"output-data-prep.csv"), index=False)

