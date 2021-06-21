import os
import argparse
import joblib

import pandas as pd
import numpy as np

from azureml.core import Run

def run(args):
    
    print('--------Inside register file --------------------')
    print(args.data_path)
    print(os.listdir(args.data_path))

    best_model = joblib.load(os.path.join(args.data_path, 'sentitect-best-model-pkl.pkl'))
    print(best_model)

    data = pd.read_csv(os.path.join(args.data_path, 'output-vectorize.csv'))
    data['vectors'] = data['vectors'].apply(lambda row: [float(t) for t in row.split(';')])

    X_train = data.vectors.values
    X_train = [np.array(tmp) for tmp in X_train]
    X_train = np.array(X_train)
    y_train = data.sentiment.values

    result = best_model.predict(X_train)
    print(np.average(result == y_train))

    runObj = Run.get_context()
    
    print('--------After getting the context --------------------')
    print(args.data_path)
    print(os.listdir(args.data_path))

    best_model = joblib.load(os.path.join(args.data_path, 'sentitect-best-model-pkl.pkl'))
    print(best_model)

    data = pd.read_csv(os.path.join(args.data_path, 'output-vectorize.csv'))
    data['vectors'] = data['vectors'].apply(lambda row: [float(t) for t in row.split(';')])

    X_train = data.vectors.values
    X_train = [np.array(tmp) for tmp in X_train]
    X_train = np.array(X_train)
    y_train = data.sentiment.values

    result = best_model.predict(X_train)
    print(np.average(result == y_train))

    #'sentitect-best-model-pkl.pkl'
    try:
        print('Files inside run context -- before download or/and upload')
        print(runObj.get_file_names())

        for tmp in runObj.get_file_names():
            print(tmp)

    except Exception as e:
        print(e)

    

    try:
        runObj.download_file(name=os.path.join(args.data_path, 'sentitect-best-model-pkl.pkl'), 
                         output_file_path='outputs/sentitect-best-model-pkl.pkl')
    except Exception as e:
        print(e)

    try:
        runObj.download_file(name=os.path.join(args.data_path, 'sentitect-best-model-pkl.pkl'), 
                         output_file_path='sentitect-best-model-pkl.pkl')
    except Exception as e:
        print(e)

    try:
        runObj.upload_file(name=os.path.join(args.data_path, 'sentitect-best-model-pkl.pkl'), 
                         path_or_stream='sentitect-best-model-pkl.pkl')
    except Exception as e:
        print(e)

    try:
        runObj.upload_file(name=os.path.join(args.data_path, 'sentitect-best-model-pkl.pkl'), 
                         path_or_stream='outputs/sentitect-best-model-pkl.pkl')
    except Exception as e:
        print(e)

    try:
        print('Files inside run context -- after download or/and upload')
        print(runObj.get_file_names())

        for tmp in runObj.get_file_names():
            print(tmp)

    except Exception as e:
        print(e)

    try:
        model = runObj.register_model(model_name='sentimodel',
                                 model_path=os.path.join(args.data_path, 'outputs/sentitect-best-model-pkl.pkl'))
    except Exception as e:
        print(e)

    try:
        model = runObj.register_model(model_name='sentimodel',
                                 model_path='outputs/sentitect-best-model-pkl.pkl')
    except Exception as e:
        print(e)

    try:
        model = runObj.register_model(model_name='sentimodel',
                                 model_path='sentitect-best-model-pkl.pkl')
    except Exception as e:
        print(e)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        dest='data_path',
        type=str,
        required=True,
        help='Path to the best model'
    )
    args = parser.parse_args()
    
    model = run(args)
    print(model.name, model.id, model.version, sep='\t')

    