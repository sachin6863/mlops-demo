from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input, Output, dsl
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes
import os

# Define Azure ML workspace info
subscription_id = '116fc71b-abc0-49b8-8d08-802c398560a1'
resource_group = 'mlops-rg'
workspace = 'mlops-ws'

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace,
)

# Environment (built-in or custom)
env = Environment(
    name="sklearn-env-3",
    image="mcr.microsoft.com/azureml/minimal-py312-inference:latest",
    conda_file='conda.yaml',
    version="1",
)

# Define the batch job as a command
batch_job = command(
    name="batch-inference",
    display_name="Batch Inference Job",
    description="Runs batch inference using a registered model",
    code="./",
    command="python score.py --input_data ${{inputs.input_data}} --output_data ${{outputs.output_data}} --model_path ${{inputs.model_path}}",
    inputs={
        "input_data": Input(path="./inference.csv", type=AssetTypes.URI_FILE),
        "model_path": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml://registries/mlops-registry/models/credit-card-default-rf/versions/1"
        )
    },
    outputs={
        "output_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")
    },
    environment=env,
    compute="mlops-cluster",
    experiment_name="batch-inference-exp"
)

# Pipeline definition
@dsl.pipeline(
    compute="mlops-cluster",
    description="Pipeline for batch inferencing"
)
def batch_inference_pipeline():
    inference_step = batch_job()
    return {"output": inference_step.outputs.output_data}

# Run the pipeline
pipeline_job = batch_inference_pipeline()
ml_client.jobs.create_or_update(pipeline_job)
