import os
import argparse
import joblib

import pandas as pd
import numpy as np

from azureml.core import Run

def run(args):
    
    print(args.model_path)
    
    runObj = Run.get_context()
    ws = runObj.experiment.workspace
    
    try:
        
        runObj.upload_file(name='sentitect-best-model-pkl.pkl', 
                         path_or_stream=os.path.join(args.model_path, 'sentitect-best-model-pkl.pkl'))

        
        model = runObj.register_model(model_name='sentimodel',
                                 model_path='sentitect-best-model-pkl.pkl',
                                 workspace=ws)
    except Exception as e:
        print(e)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        dest='model_path',
        type=str,
        required=True,
        help='Path to the best model'
    )
    args = parser.parse_args()
    
    model = run(args)
    print(model.name, model.id, model.version, sep='\t')

    