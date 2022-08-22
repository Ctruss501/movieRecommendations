import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template
import re
import json


# Importing preprocessed movie dataset. Creating count matrix and similarity score matrix.
def create_model():
    data = pd.read_csv('final_movie_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['combined_features'])
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(count_matrix)
    return data, model, count_matrix


# Function to find similar movies for the movie that was searched. First movie to be
# returned is the searched movie and the next 15 are recommended/similar movies.
def recommend(choice, model=None):
    try:
        model.get_params()
    except:
        data, model, count_matrix = create_model()
    if choice in data['combined_title'].values:
        choice_index = data[data['combined_title'] == choice].index.values[0]
        distances, indices = model.kneighbors(count_matrix[choice_index], n_neighbors=16)
        movie_list = []
        for i in indices.flatten():
            movie_list.append(data[data.index == i]['title'].values[0].title())
        return movie_list
    elif (data['combined_title'].str.contains(choice).any() == True):
        similar_names = list(str(s) for s in data['combined_title'] if choice in str(s))
        similar_names.sort()
        new_choice = similar_names[0]
        print(new_choice)
        choice_index = data[data['combined_title'] == new_choice].index.values[0]
        distances, indices = model.kneighbors(count_matrix[choice_index], n_neighbors=16)
        movie_list = []
        for i in indices.flatten():
            movie_list.append(data[data.index == i]['title'].values[0].title())
        return movie_list
    else:
        return "Sorry, this movie is not in our database."

# Function to check the titles in the data file while user is typing.
# A list will start to appear under search bar with titles similar to
# what the user is typing to help ensure one is selected from the dataset,
def get_suggestions():
    data = pd.read_csv('final_movie_data.csv')
    return list(data['title'].str.capitalize())


app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('main.html', suggestions=suggestions)


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = recommend(movie)
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str


@app.route("/Search")
def search_movies():
    suggestions = get_suggestions()
    choice = request.args.get('movie')
    choice = re.sub("[^a-zA-Z1-9]", "", choice).lower()
    movies = recommend(choice)
    if type(movies) == type('string'):
        return render_template('movies.html', movie=movies, s='Sorry')
    else:
        return render_template('movies.html', movie=movies)


if __name__ == "__main__":
    app.run(debug=True)
