from azureml.pipeline.core import PublishedPipeline
from azureml.pipeline.core import PipelineEndpoint

from azureml.core import Experiment


from azureml.core import Workspace
import requests

ws = Workspace.from_config()
print(ws)

published_pipeline = PublishedPipeline.get(workspace=ws, id="f8d8a01b-57a9-4d53-bd50-f4eef67f610d")
print(published_pipeline)

'''


ws1 = Workspace(subscription_id="2a312bc4-2d2c-41be-810d-95fbe06a6aa3",
               resource_group="sentitect",
               workspace_name="sentitect-ws")
print(ws1)

published_pipeline = PublishedPipeline.get(workspace=ws, id="f8d8a01b-57a9-4d53-bd50-f4eef67f610d")
print(published_pipeline)

response = requests.post(published_pipeline.endpoint,
                         headers={'Content-Type':'application/json'},
                         json={"ExperimentName": "my_http_experiment"})
                         #,"ParameterAssignments": {"pipeline_arg": 20}})

print(response)
'''

'''
experiment = Experiment(workspace=ws, name='my_experiment')
pipeline_run = experiment.submit(published_pipeline,
                              pipeline_parameters={})

print(pipeline_run)

'''

'''

# tenant id - 5254eecf-63bd-403a-9077-71f987bb793b
# workspace suscription id - 2a312bc4-2d2c-41be-810d-95fbe06a6aa3
# pipeline id - f8d8a01b-57a9-4d53-bd50-f4eef67f610d

# api id - 3f6dff05-a9d0-4d1d-8607-4e029e69c546

from azureml.core.authentication import TokenAuthentication, Audience

# This is a sample method to retrieve token and will be passed to TokenAuthentication
def get_token_for_audience(audience):
    from adal import AuthenticationContext
    client_id = "3f6dff05-a9d0-4d1d-8607-4e029e69c546"
    client_secret = "YxZ5Du_3L4_lx~WfzjmZ0SEZ6tdweeI7-q"
    tenant_id = "5254eecf-63bd-403a-9077-71f987bb793b"
    auth_context = AuthenticationContext("https://login.microsoftonline.com/{}".format(tenant_id))
    resp = auth_context.acquire_token_with_client_credentials(audience,client_id,client_secret)
    token = resp["accessToken"]
    return token


token_auth = TokenAuthentication(get_token_for_audience=get_token_for_audience)

print(token_auth)

print(Audience.ARM)
print(Audience.AZUREML)

token_arm_audience = token_auth.get_token(Audience.AZUREML)
token_aml_audience = token_auth.get_token(Audience.ARM)

print(token_aml_audience)
print('----------')
print(token_arm_audience)


'''
aad_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Im5PbzNaRHJPRFhFSzFqS1doWHNsSFJfS1hFZyIsImtpZCI6Im5PbzNaRHJPRFhFSzFqS1doWHNsSFJfS1hFZyJ9.eyJhdWQiOiJodHRwczovL21sLmF6dXJlLmNvbSIsImlzcyI6Imh0dHBzOi8vc3RzLndpbmRvd3MubmV0LzUyNTRlZWNmLTYzYmQtNDAzYS05MDc3LTcxZjk4N2JiNzkzYi8iLCJpYXQiOjE2MjQ1MTM3MDMsIm5iZiI6MTYyNDUxMzcwMywiZXhwIjoxNjI0NTE3NjAzLCJhaW8iOiJFMlpnWVBpdE9OUG9sYzdqbjljbSthMFE4OGs5RGdBPSIsImFwcGlkIjoiM2Y2ZGZmMDUtYTlkMC00ZDFkLTg2MDctNGUwMjllNjljNTQ2IiwiYXBwaWRhY3IiOiIxIiwiaWRwIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvNTI1NGVlY2YtNjNiZC00MDNhLTkwNzctNzFmOTg3YmI3OTNiLyIsIm9pZCI6ImU4MjdjZmNjLTk1YTUtNDUxMi05MzFjLTM5ZDhmMTllZWI2MiIsInJoIjoiMC5BWUVBei01VVVyMWpPa0NRZDNINWg3dDVPd1hfYlRfUXFSMU5oZ2RPQXA1cHhVYUJBQUEuIiwic3ViIjoiZTgyN2NmY2MtOTVhNS00NTEyLTkzMWMtMzlkOGYxOWVlYjYyIiwidGlkIjoiNTI1NGVlY2YtNjNiZC00MDNhLTkwNzctNzFmOTg3YmI3OTNiIiwidXRpIjoiSnhPaG9FZ2wzMEthWXhWM2ZRUllBUSIsInZlciI6IjEuMCJ9.Ki2XDzD11KS1t_GCM8nxfO7pfXBrKbE0C0-uOzD5cEwNgEqpz3rNAv8qvqLF1iTe28Z57JAdr0OCwFKTRQNUSPSgd3NQUhff_9VAhNtpiP6N9v00iEeHxT4riF5XzrLu2Z0p3lZsdHoGsfhoGMeJqTaOjFJufJEPc1PpxnLN83kns0V0tXwIgvNJCPZ28HKDETcOxcCfH23VrKxVtw-WyvRtgDu3TYtQcncZVAtT5BVSytoDHwRqrzEIktB0P5fQLuGLOCn9OaSl9vyUFoiQKUe15EVHORzAPRKE8YP_xSzTOyJpA10tDBqX9twLXiHnx9EMixBAMjLCPV1kb5RH6A'

headers = {'Content-Type': 'application/json'}


response = requests.post(published_pipeline.endpoint,
                         headers=headers,
                         json={"ExperimentName": "my_http_experiment",
                         "ParameterAssignments": {"pipeline_arg": 20}})


print(response)


'''

pipeline_endpoint = PipelineEndpoint.publish(workspace=ws, name="PipelineEndpointTest",
                                            pipeline=published_pipeline, description="Test description Notebook")



pipeline_endpoint_by_name = PipelineEndpoint.get(workspace=ws, name="PipelineEndpointTest")
run_id = pipeline_endpoint_by_name.submit("PipelineEndpointExperiment")
print(run_id)


rest_endpoint = pipeline_endpoint_by_name.endpoint
response = requests.post(rest_endpoint, 
                         headers=aad_token, 
                         json={"ExperimentName": "PipelineEndpointExperiment",
                               "RunSource": "API",
                               "ParameterAssignments": {"1": "united", "2":"city"}})


response = requests.post(published_pipeline.endpoint,
                         #headers=aad_token,
                         json={"ExperimentName": "check_deploy"})
                         #,"ParameterAssignments": {"pipeline_arg": 20}})


print(response)

rest_endpoint = 'https://westeurope.api.azureml.ms/pipelines/v1.0/subscriptions/2a312bc4-2d2c-41be-810d-95fbe06a6aa3/resourceGroups/sentitect/providers/Microsoft.MachineLearningServices/workspaces/sentitect-ws/PipelineRuns/PipelineSubmit/f8d8a01b-57a9-4d53-bd50-f4eef67f610d'

response = requests.post(rest_endpoint, 
                         headers=aad_token, 
                         json={"ExperimentName": "PipelineEndpointExperiment",
                               "RunSource": "API",
                               "ParameterAssignments": {"1": "united", "2":"city"}})

'''
