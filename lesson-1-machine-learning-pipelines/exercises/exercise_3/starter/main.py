import mlflow
import os
import hydra
from mlflow.utils import uri 
# package for Facebook research for configuring complex applications
# pip install hydra-core --upgrade
from omegaconf import DictConfig
# pip install --upgrade omegaconf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        # URI can be a local path or the URL to a git repository
        # here we are going to use a local path
        uri = os.path.join(root_path, "download_data"),
        entry_point = "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        }
    )

    ##################
    # Your code here: use the artifact we created in the previous step as input for the `process_data` step
    # and produce a new artifact called "cleaned_data".
    # NOTE: use os.path.join(root_path, "process_data") to get the path
    # to the "process_data" component
    ##################

    _ = mlflow.run(
        # URI can be a local path or the URL to a git repository
        # here we are going to use a local path
        uri = os.path.join(root_path, "process_data"),
        entry_point = "main",
        parameters={
            "input_artifact": "iris.csv:latest",
            "artifact_name": "clean_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Clean data"
        }
    )

if __name__ == "__main__":
    go()
