from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "sentiment",  "method" : "lightgbm"}, 
                                               description='Predict sentiments of movie reviews')


from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace.from_config()
model = Model(ws, 'sentitect_lightgbm')

env = Environment('sentitect-env')
cd = CondaDependencies(conda_dependencies_file_path='../environment.yml')
env.python.conda_dependencies = cd
inference_config = InferenceConfig(entry_script="deploy.py", environment=myenv)

service = Model.deploy(workspace=ws, 
                       name='sentitect_lightgbm', 
                       models=[model], 
                       inference_config=inference_config, 
                       deployment_config=aciconfig)

service.wait_for_deployment(show_output=True)

print(service.scoring_uri)