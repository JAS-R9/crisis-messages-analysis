# Disaster Response Pipeline Project
This repository has a trained model to analyze messages for disaster response and classify them to their relevant categories.  

# Libraries Used:
1. numpy
2. pandas
3. sqlalchemy
4. nltk
5. pickle
6. re
7. sklearn

# installation:
No installtion required. However, follow the steps below to use the model or retrain based on your data.
1. Run the following command in the app's directory to run your web app. 
	- `python run.py`
    
2. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to http://0.0.0.0:3001/

# Files in the repository:
## data folder:
### disaster_categoires.csv and disaster_messages.csv
 These files include the data used to train the model. 
### process_data.py
This ETL program will load two files, transform them, and load the prepared version to a database
Parameters:
    argument 1: Messages CSV file
    argument 2: Categories CSV file  
Returns:
    .db file: a database file containing the cleaned data
### DisasterResponse.db
This is the database containing the result of running the ETL process_data.py on the two csv files

## models folder
### train_classifier.py
This program will use a set of existing messages stored in a database to develop a ML model based on how the messages are classified. Then the  model will be stored in a pickle file that should be specified in the arguments. 
Parameters:
	argument 1: Database file name
    arugment 2: Pickle file name
Returns:
    GridSearchCV: trained model 
### classifier.pkl
This pickle file contains the trained model

## app folder
### run.py
This is the python file that will run the website and contain the visualization available in the website.
### templates folder
This folder contains go.html and master.html which are the two main HTML components of the website. 
