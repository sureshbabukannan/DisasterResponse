import json
import plotly
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
#    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    stop_words = stopwords.words("english")
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)    
        clean_tokens = [t for t in clean_tokens if t not in stop_words]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
#df.dropna(axis='index',inplace=True)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Graph #1 data - Count by Genre of message 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Graph #2 data Count of each catergories
    df_cat = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 
       'search_and_rescue', 'security', 'military', 'child_alone', 
       'water', 'food', 'shelter', 'clothing', 'money', 
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 
       'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 
       'other_infrastructure', 'weather_related', 'floods', 'storm', 
       'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'
    ]]
    
    for y in df_cat.columns:
        df_cat[y] = df_cat[y].astype(int)
    
    category_values = df_cat.sum().sort_values(ascending=False).values
    category_names = df_cat.sum().sort_values(ascending=False).index
    
    # Graph #3 data - Count of unique word in all the messages
    # Pick induvidual words from all of the messages.
    a = df['message'].apply(tokenize)
    b = a.apply(pd.Series).stack().reset_index(drop=True)
    list_explode = pd.DataFrame(b, columns=['word'])
    word_counts = list_explode.groupby('word')['word'].count().sort_values(ascending=False)

    # There are too many words only pick words which are more than 400 occurances.
    word_count_values = word_counts[word_counts > 400].values
    words = word_counts[word_counts > 400].index

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data' :[
                Bar(
                    x=category_names,
                    y=category_values
                )
            ],

            'layout': {
                'title': 'Distriution of categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=words,
                    y=word_count_values
                )
            ],
            'layout': {
                'title': 'Word Count of words in messages occuring 400 times',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()