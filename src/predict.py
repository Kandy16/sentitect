import os
import argparse

import numpy as np
import pandas as pd

from azureml.core import Run
from azureml.core.runconfig import DataPath
from azureml.core.model import Model
import joblib

def run(args):

    runObj = Run.get_context()
    ws = runObj.experiment.workspace

    reg_model_path = Model.get_model_path(model_name='sentimodel', _workspace=ws)
    #print(os.getenv('AZUREML_MODEL_DIR'))
    #reg_model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sentimodel')
    best_model = joblib.load(reg_model_path)
    
    # read the vectors an convert them to numpy
    data = pd.read_csv(os.path.join(args.data_path,"output-vectorize.csv"))
    data['vectors'] = data['vectors'].apply(lambda row: [float(t) for t in row.split(';')])
    x = data.vectors.values
    x = [np.array(tmp) for tmp in x]
    x = np.array(x)

    predictions = best_model.predict(x)
    data['prediction'] = predictions
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path',
        dest='data_path',
        type=str,
        required=True,
        help='Path to the predict data'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output result path'
    )        
    args = parser.parse_args()
    
    result = run(args)
    result[['review','prediction']].to_csv(os.path.join(args.output, 'output.csv'), index=False)
    print(result.head(5))
     
