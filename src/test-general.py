from azureml.pipeline.core import PublishedPipeline
from azureml.pipeline.core import PipelineEndpoint
from azureml.core import Workspace
import requests

ws = Workspace.from_config()
print(ws)


published_pipeline = PipelineEndpoint.get(workspace=ws, name="check_deploy")
print(published_pipeline)