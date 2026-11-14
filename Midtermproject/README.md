# Midterm Project
## Description of the problem
In this midterm project the aim is to build a model that predict whether a patient is either a no-show or will attend their medical appointment. 

### Problem:
Patients make an appointment without showing up, wasting recourses of the healthcare providers

### Solution:
Make a model that predicts whether a patient will attend their appointment or not, based on a patients demographics, time between appointments, and other available characteristics.

## Instructions on how to run the project
The notebook.ipynb file contains all of the data exploration and model training. 
train.py is the exported notebook which has been cleaned up afterwards, containing only the training and validation of the best model including training the final model. 

Data is available in the 'data' folder, both the notebook and the train.py script should automatically read in the dataset.
If you are unable to execute it due to data not working the original dataset can be found on kaggle (https://www.kaggle.com/datasets/joniarroba/noshowappointments/data)

### Dependency and environment management
The project uses pipenv for dependency and environment management.
The dependencies and versions are defined in 'Pipfile' and 'Pipfile.lock'. 

if needed install Pipenv:
pip install pipenv

### Instruction on how to set up the environment and run the model
From the project directory, install the dependencies:
Make sure you are in the Midtermproject directory.
run 'cd Midtermproject' if you are in the toplevel folder

Install the dependencies:
- pipenv install

Activate the virtual environment:
- pipenv shell 

The model is set up as a Flask web service. It can be started by running the following line in a terminal:
- waitress-serve --listen=0.0.0.0:9696 predict:app

to test the model prediction open another terminal and run the follwoing lines:
- pipenv shell
- python predict-test.py
