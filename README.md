## Create conda environment

Checkout the [link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) to see how to create a conda environment.

```
conda create -n <env_name> python=3.9
```
Activate the conda environment

```
conda activate <env_name>
```

## Install the python packages

```
pip install -r requirements.txt
```

## Run the mlflow server
```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
