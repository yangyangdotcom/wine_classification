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

## Track the experiments using mlflow

Start the mlflow server
```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
Run the experiments
```
python train_dl.py
```
OR
```
python train_svc.py
```
Access the localhost to view all the experiment performances and more

## Prefect
Prefect is being used here for managing the workflow execution, ensuring tasks are run in the correct order and handling any failure

We can start the prefect server by running
```
prefect server start
```
Access the localhost after running any train python files and you can see the flow runs

## Deploy using Flask
predict.py serves as a RESTful API for serving predictions from a machine learning model trained with MLflow. We can start it by running 
```
python predict.py
```
We can test the api by using  test_flask.py. Simply run the following command in a new terminal
```
python test_flask.py
```

## Docker
Dockerfile provides a standardized and reproducible way to package and deploy applications as Docker containers. We can build the docker image by running:
```
docker build -t <docker_name> .
```
Once the image is built, we can run a container based on it using:
```
docker run -p 9696:9696 my_mlflow_app
```