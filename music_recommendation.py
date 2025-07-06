from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Data preparation
songs = pd.read_csv('/Users/dhruvidesai/Downloads/ex.csv')

# Data Cleaning
songs.dropna(inplace=True)
songs = songs.drop_duplicates()

songs['User-Rating'] = songs['User-Rating'].apply(lambda x: x[:3])
songs['Album/Movie'] = songs['Album/Movie'].str.replace(' ', '', regex=False)
songs['Singer/Artists'] = songs['Singer/Artists'].str.replace(' ', '',
                                                              regex=False)
songs['Singer/Artists'] = songs['Singer/Artists'].str.replace(',', ' ',
                                                              regex=False)

songs['tags'] = (
        songs['Singer/Artists'] + ' ' +
        songs['Genre'] + ' ' +
        songs['Album/Movie'] + ' ' +
        songs['User-Rating']
)
songs['tags'] = songs['tags'].apply(lambda x: x.lower())

new_songs = songs[['Song-Name', 'tags']].rename(columns={'Song-Name': 'title'})

# Text Vectorization and Similarity Matrix
cv = CountVectorizer(max_features=2000)
vectors = cv.fit_transform(new_songs['tags']).toarray()
similarity = cosine_similarity(vectors)

# Save preprocessed data
pickle.dump(new_songs, open('musicrec.pkl', 'wb'))
pickle.dump(similarity, open('similarities.pkl', 'wb'))


# Recommendation function
def recommend(music, num_recommendations):
    try:
        music_index = new_songs[new_songs['title'] == music].index[0]
        distances = similarity[music_index]
        music_list = sorted(list(enumerate(distances)), reverse=True,
                            key=lambda x: x[1])[1:num_recommendations + 1]

        recommendations = []
        for i in music_list:
            recommendations.append(new_songs.iloc[i[0]].to_dict())
        return recommendations
    except IndexError:
        return {
            "error": f"'{music}' not found in the dataset. Please check the spelling or try another song."}


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/recommend', methods=["POST"])
def get_recommendations():
    data = request.json
    song_title = data.get("song_title")
    num_recommendations = data.get("num_recommendations", 3)

    recommendations = recommend(song_title, num_recommendations)
    return jsonify({"recommendations": recommendations})


@app.route('/songs', methods=["GET"])
def get_songs():
    song_titles = new_songs['title'].tolist()
    return jsonify({"songs": [{"title": title} for title in song_titles]})


if __name__ == "__main__":
    app.run(debug=True)
