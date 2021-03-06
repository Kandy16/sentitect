import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.runconfig import RunConfiguration

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

from azureml.pipeline.steps import PythonScriptStep
from azureml.data import OutputFileDatasetConfig
from azureml.data.datapath import DataPath

#from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

import os

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, sep='\t')


aml_run_config = RunConfiguration()

# choose a name for your cluster. If not available then create the instance
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "senti-predict")
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

    # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                min_nodes=compute_min_nodes,
                                                                max_nodes=compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(
        ws, compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

# For a more detailed view of current AmlCompute status, use get_status()
print(compute_target.get_status().serialize())

# assign target to run_config
aml_run_config.target = compute_target

# to install required packages
env = Environment('sentitect-env')
cd = CondaDependencies(conda_dependencies_file_path='../environment.yml')
env.python.conda_dependencies = cd
 
aml_run_config.environment = env

#first of all the local data folder is copied into a datastore.
datastore = ws.get_default_datastore()
datastore.upload(src_dir='../data-predict',
                 target_path='data-predict/',
                 overwrite=True)


datastore_path = [
    DataPath(datastore, 'data/predict-data.csv')
]
#dataset = Dataset.File.from_files(path=datastore_path)
dataset = Dataset.File.from_files(path=(datastore, 'data-predict/'))

output_data1 = OutputFileDatasetConfig(destination = (datastore, 'outputdataset/{run-id}'))

# the data preparation does primitive cleaning and stores the intermediate result in datastore
#data-count -1 will read all the data. 100 or 200 size is considered a toy set
data_prep_step = PythonScriptStep(
    source_directory='.',
    script_name='data_prep.py',
    arguments=[
        '--data-path', dataset.as_named_input('input').as_mount(),
        '--data-count',10,
        "--output", output_data1
        ],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

# vectorization step takes the previous intermediate result and uses spacy nlp pipeline to compute vectors of sentences
# the intermediate result is stored in datastore
output_data2 = OutputFileDatasetConfig(destination = (datastore, 'outputdataset/{run-id}'))

vectorize_step = PythonScriptStep(
    source_directory='.',
    script_name='vectorize.py',
    arguments=[
        '--data-path', output_data1.as_input(),
        "--output", output_data2
        ],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

# training step involves multiple models and compare their performance of validation dataset
# It uses randomsearch and gridsearch of parameters and compares 100 models
# the efficient model is saved to a persistent location
output_data3 = OutputFileDatasetConfig(destination = (datastore, 'outputdataset/{run-id}'))

predict_step = PythonScriptStep(
    source_directory='.',
    script_name='predict.py',
    arguments=[
        '--data-path', output_data2.as_input(),
        '--output', output_data3
        ],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)
 
# the pipeline sequence
train_models = [data_prep_step, vectorize_step, predict_step]

from azureml.pipeline.core import Pipeline

# Build the pipeline
pipeline1 = Pipeline(workspace=ws, steps=[train_models])

# Submit the pipeline to be run
experiment_name = 'sentitect-expr'
pipeline_run1 = Experiment(workspace=ws, name=experiment_name).submit(pipeline1)
pipeline_run1.wait_for_completion()
