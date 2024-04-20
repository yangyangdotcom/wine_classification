# Use a Miniconda base image
FROM continuumio/miniconda3:latest

# Install gcc and other essential tools
RUN apt-get update && \
    apt-get install -y build-essential

# Set the working directory in the container
WORKDIR /app

# Install your primary dependencies directly
# Note: You will need to replace 'package_name' with actual package names and versions
RUN conda install -c conda-forge python=3.9 flask=3.0.2 gunicorn=20.1.0 numpy=1.22.4 scikit-learn=1.0.2 pandas=2.2.2 mlflow=2.11.3

# Optionally, use pip for any packages not available or up-to-date in Conda
# COPY requirements.txt /app/
# RUN pip install -r requirements.txt

# Copy the rest of your application files to the container
COPY [ "/home/ubuntu/wine_classification/model", "predict.py", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Specify the command to run your application
ENTRYPOINT ["gunicorn", "--bind=0.0.0:9696", "predict:app"]