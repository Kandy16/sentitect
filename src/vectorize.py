import os
import argparse

import numpy as np
import pandas as pd
import spacy

def main(data):

    print(data.head(5))
    #VECTORIZE the sentence using spacy
    nlp = spacy.load('en_core_web_sm')

    # Embed sentences for the training set
    # It took a lot of time when used all the pipeline chains of space
    # Disable irrelevant sub-pipelines. The quality of vectors may get affected -- need to test
    X_train = []

    '''    
    for r in nlp.pipe(data.review.values, disable=['parser','ner','entity_linker','entity_ruler',
                    'textcat','textcat_multilabel','lemmatizer', 'morphologizer',
                    'attribute_ruler','senter','sentencizer','tok2vec','transformer']):
    '''
    for r in nlp.pipe(data.review.values):
        
        emb = r.vector
        review_emb = emb.reshape(-1)
        X_train.append(review_emb)

    X_train = np.array(X_train)
    y_train = data.sentiment.values if 'sentiment' in data.columns else None
    
    return X_train,y_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )

    args = parser.parse_args()
    result = main(args)
    print(result[0])