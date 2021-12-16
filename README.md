# Disaster_Response
A machine learning pipeline that categorizes messages that are sent during disaster events so that the messages can be set to an appropriate disaster relief agency.
## 1. Installation:
Scikit-learn library is requires for running this project.The easiest way to install scikit-learn is using:
  - pip (pip install -U scikit-learn) 
  - or conda (conda install -c conda-forge scikit-learn)
All  other libraries that are used in this project are installed with the Anaconda distribution.
Python version: Python 3.8.8

## 2. Project Motivation:

During disasters events, many people are in need of help from different relief agencies. This is a motivation to create a web app where people can send messages that explain their needs. The web app then categorizes the received messages and sends them to the appropriate disaster relief agency. This will accelerate the operation of relieving people.  

## 3.Files Descriptions:

#### data:
##### - disaster_categories.csv: 
The source of the dataset.
##### - disaster_messages.csv: 
The source of the dataset.
##### - process_data.py:
A python file with the ETL pipline.
##### - DisasterResponse.db: 
The database of the clean data.
### model: 
##### - train_classifier.py: 
A python file with the ML piline.
### app:
##### - template: 
- master.html: the main page of web app
- go.html: the classification result page of web app
##### - run.py
A Flask file that runs app

## 4. Project Details:
1. Run the following commands in the project's root directory to set up the database and model.
    - To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app.
    python app/run.py

## 5. Author:
This project was created by Ghofran Al Tawfiq
