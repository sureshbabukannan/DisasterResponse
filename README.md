# Disaster Response Pipeline Project
During a disaster like floods, earthqukes time there could be thousands of distress message originating from social media. It would be help if the messages conext can be understood and categories
to eliminate the need for human to full read through the message to understand the conext and mannually categories it.  This will enable resuce and relief groups to focus their attention on messages relevent to them based on the category of messages.

This project aims at categoriesing a given distrest message into multiple categories. A NLP text learning ML model categories the given text on web page into multiple categories. 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database. You the find the process_data.py script file with data files disaster_messages.csv and disaster_categories.csv in ./disaster_response_pipeline_project/data folder. Run the command in the same folder.
        `python process_data.py disaster_messages.csv disaster_categories.csv MessageCategory.db`
    - To run ML pipeline that trains classifier and saves model in pickle file
        `python models/train_classifier.py data/MessageCategory.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project Files

.\Data - folder files
1. process_data.py -  This is the python script to load data into Pandas dataframes, merge the message and categories data into single dataframe and load the cleaned data to SQL lite database. 
2. messages.csv    -  This is the comma sepereated file of messages and classification of message category. This data is merged with categories.csv data by running  process_data.py
3. categories.csv  -  This is the comma sepereated file of messages classification with each classification coded with caterogy value-0/1.  This data is cleaned in process_data.py
4. MessageCategory.db - This is the SQLite Database create following the executing of process_data.py the cleaned data is loaded into the database.


.\models
1. train_classifier.py - The python script to build, train , evaluate and save model. The scripts uses data from MessageCategory.db SQLite database
2. classifier.pkl - the model file outout from train_classifier.py script. 
3. environment.yaml - the yaml file to create runtime environment for the train_classifier.py to run. The conda enviroment used to test the ML script is on WIN10 64bit system

### ETL Data Processing
The CSV data files are loaded to pandas dataframes to clean for duplicates, catergory dummy variable columns are added to enable machine learning. The cleaned data is loaded to SQLite database.

### Machine Learning Pipeline  
The Randomforest classifier is used as input to Multiclassoutput classifer to fit and predict multiple labels.
The message is the column used for subject of training which have to multiple labelled for 36 labels. a pipeline with Countvectorizer and Tfidftransform and Multiclassoutput classifer is created. The 
GridSearchCV is used with mutiple parameters for chossing best parameters for transformer and estimator steps. 

### Web App
