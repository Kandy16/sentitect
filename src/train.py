import os
import argparse

import numpy as np
import pandas as pd
import re

import spacy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import lightgbm as lgbm

import time


start_time = time.time()




from azureml.core import Run
run = Run.get_context()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for SGD'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD'
    )
    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    # LOAD DATA SETS
# Load the train, test and submission data frames

    TOY_NUM = 200

    print('--------------- Data preparation ------------------')

    train_df = pd.read_csv(os.path.join(args.data_path,"train-data.csv"))
    train_df = shuffle(train_df)[0:TOY_NUM]

    test_df = pd.read_csv(os.path.join(args.data_path,"test-data.csv"))
    test_df = shuffle(test_df)[0:TOY_NUM]

    submission_df = pd.read_csv(os.path.join(args.data_path,"predict-data.csv"))
    submission_df = shuffle(submission_df)[0:TOY_NUM]

    # Create a merged data set and review initial information
    combined_df = pd.concat([train_df, test_df])

    # DATA EXPLORATION

    # Quickly check for class imbalance
    print(combined_df.describe())

    # Check what the text looks like
    print(combined_df.head(5))

    # Get all the unique keywords
    #print(combined_df["review"]-str.split.unique())

    # Create small function to clean text
    def text_clean(text):

        for element in ["http\S+", "RT ", "[^a-zA-Z\'\.\,\d\s]", "[0-9]","\t", "\n", "\s+", "<.*?>"]:
            text = re.sub("r"+element, " ", text)

        return text

    # Clean data sets
    combined_df.review = combined_df.review.apply(text_clean)
    #test_df.review = test_df.review.apply(text_clean)
    submission_df.review = submission_df.review.apply(text_clean)

    # CORRECT SPELLING

    print('--------------- Vectorizing data ------------------')

    start_vector_time = time.time()

    #VECTORIZE the sentence
    nlp = spacy.load('en_core_web_sm')

    # Embed sentences for the training set
    X_train = []
    for r in nlp.pipe(combined_df.review.values, disable=['parser','ner','entity_linker','entity_ruler',
                    'textcat','textcat_multilabel','lemmatizer', 'morphologizer',
                    'attribute_ruler','senter','sentencizer','tok2vec','transformer']):

        #print(f"{idx} out of {train_df.shape[0]}")
        emb = r.vector
        review_emb = emb.reshape(-1)
        X_train.append(review_emb)

    X_train = np.array(X_train)
    y_train = combined_df.sentiment.values

    end_vector_time = time.time()

    print(f'Vectorization Time taken in seconds : {end_vector_time - start_vector_time}')
    print(f'Vectorization Time taken in minutes : {(end_vector_time - start_vector_time)/60}')

    '''

    # Embed sentences for the submission set
    submission_data = []
    for r in nlp.pipe(submission_df.review.values):
        emb = r.vector
        review_emb = emb.reshape(-1)
        submission_data.append(review_emb)

    submission_data = np.array(submission_data)

    '''
    print('--------------- Training Data ------------------')


    # LGBM

    # Split data into train and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2, random_state = 42)

    # Get the train and test data for the training sequence
    train_data = lgbm.Dataset(X_train, label=y_train)
    test_data = lgbm.Dataset(X_test, label=y_test)

    # Parameters we'll use for the prediction
    parameters = {
        'application': 'binary',
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'dart',
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 0
    }

    # Train the classifier
    classifier = lgbm.train(parameters,
                        train_data,
                        valid_sets= test_data,
                        num_boost_round=5000,
                        early_stopping_rounds=100)

'''

print('--------------- Prediction ------------------')

# PREDICTION
val_pred = classifier.predict(submission_data)

# Submission file
submission_df['sentiment_predicted'] = val_pred.round().astype(int)
submission_df.to_csv('submission_lgbm.csv', index=False)

'''
end_time = time.time()

print(f'Time taken in seconds : {end_time - start_time}')
print(f'Time taken in minutes : {(end_time - start_time)/60}')


#correct_pred_count =  sum(submission_df['sentiment'] == submission_df['sentiment_predicted'])
#print('The accuracy is : ', (100*correct_pred_count/submission_df.shape[0]))

print('Finished Training')