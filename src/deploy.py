import json
import numpy as np
import os
import pickle
import joblib

from azureml.core import Workspace
from azureml.core import Dataset
from azureml.core.runconfig import DataPath

def init():
    global model
    
    # copy the model to local space and build it and run it
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/best-model/'))

    dataset.download(target_path='../temp/',
                 overwrite=True)
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join('../temp', 'sentitect-best-model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()
