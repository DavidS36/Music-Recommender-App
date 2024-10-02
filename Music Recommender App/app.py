from flask import Flask, request, render_template, jsonify
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np

app = Flask(__name__)

# Load data function assuming CSV format
def load_data(filepath):
    return pd.read_csv(filepath)

# Load your data once when the app starts
bp_tabular = load_data('data/bp_tabular.csv')
bp_song = load_data('data/bp_song.csv')
sample_uuid = load_data('data/sample_uuid.csv')

image_features_list = np.load('data/image_features_list.npy')
image_track_ids_list = np.load('data/image_track_ids_list.npy')

audio_features_list = np.load('data/audio_features_list.npy')
audio_track_ids_list = np.load('data/audio_track_ids_list.npy')


# Function for Tabular Data
def find_similar_song(input_song, song_data, n_recommendations=1):
    input_features = song_data[song_data['track_id'] == input_song][['bpm', 'key_id']].values[0]
    song_data['distance'] = song_data.apply(
        lambda row: euclidean(input_features, row[['bpm', 'key_id']].values), axis=1
    )
    recommendations = song_data.sort_values('distance').head(n_recommendations)
    return recommendations[['track_id', 'song', 'bpm', 'key_id']]

# Tabular Data function
def recommend_songs(input_song, song_data, n_recommendations=5):
    input_cluster = song_data[song_data['track_id'] == input_song]['Cluster'].values[0]
    same_cluster_songs = song_data[song_data['Cluster'] == input_cluster]
    recommendations = find_similar_song(input_song, same_cluster_songs, n_recommendations=n_recommendations)
    return recommendations

# Function for Image Data and Audio Data
def compute_similarity(features_list, input_feature_vector, metric='euclidean'):
    if metric == 'euclidean':
        distances = [euclidean(input_feature_vector, feature) for feature in features_list]
    return np.array(distances)

# Image Data function
def recommender_image(input_track_id, features_list, track_ids_list, bp_song, top_n=5):
    input_index = np.where(track_ids_list == input_track_id)[0][0]
    input_feature_vector = features_list[input_index]
    similarities = compute_similarity(features_list, input_feature_vector, metric='euclidean')
    most_similar_indices = similarities.argsort()[1:top_n + 1]  # Exclude the input track itself
    recommended_track_ids = [track_ids_list[idx] for idx in most_similar_indices]
    recommended_songs = bp_song[bp_song['track_id'].isin(recommended_track_ids)]
    return recommended_songs[['track_id', 'song']]

# Audio Data function
def recommender_audio(input_track_id, features_scaled, track_id_list, bp_song, top_n=5):
    input_index = np.where(track_id_list == input_track_id)[0][0]
    input_feature_vector = features_scaled[input_index]
    similarities = compute_similarity(features_scaled, input_feature_vector, metric='euclidean')
    most_similar_indices = similarities.argsort()[1:top_n + 1]  # Exclude the input track itself
    recommended_track_ids = [track_id_list[idx] for idx in most_similar_indices]
    recommended_songs = bp_song[bp_song['track_id'].isin(recommended_track_ids)]
    return recommended_songs[['track_id', 'song']]


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = int(request.form['user_input'])  # Track ID is still passed
    recommender = request.form['recommender']  # Selected recommender type

    # Validate that the track ID exists in the dataset
    if user_input not in bp_song['track_id'].values:
        error_message = "This is not a valid entry. Please try again with a different track ID."
        return render_template('error.html', error_message=error_message)

    # Fetch the input song information
    input_song_info = bp_song[bp_song['track_id'] == user_input]['song'].values[0]
    input_sample_uuid = sample_uuid[sample_uuid['track_id'] == user_input]['sample_uuid'].values[0]
    input_audio_path = f"static/audio_files/{input_sample_uuid}.mp3"

    color_mapping = {
        'Tabular Data': '#8AC926',
        'Image Data': '#1982C4',
        'Audio Data': '#FF595E'
    }

    # Check which recommender was selected and provide the respective recommendations
    if recommender == 'Tabular Data':
        recommendations = recommend_songs(user_input, bp_tabular, 5)
        recommendations['audio_path'] = recommendations['track_id'].apply(
            lambda track_id: f"static/audio_files/{sample_uuid[sample_uuid['track_id'] == track_id]['sample_uuid'].values[0]}.mp3")
        recommendations_list = recommendations[['track_id', 'song', 'audio_path']].to_dict(orient='records')
        return render_template('results_tabular.html', input_song=input_song_info, recommendations=recommendations_list, track_id=user_input, color=color_mapping['Tabular Data'])

    elif recommender == 'Image Data':
        recommendations = recommender_image(user_input, image_features_list, image_track_ids_list, bp_song, top_n=5)
        recommendations['audio_path'] = recommendations['track_id'].apply(
            lambda track_id: f"static/audio_files/{sample_uuid[sample_uuid['track_id'] == track_id]['sample_uuid'].values[0]}.mp3")
        recommendations_list = recommendations[['track_id', 'song', 'audio_path']].to_dict(orient='records')
        return render_template('results_image.html', input_song=input_song_info, recommendations=recommendations_list, track_id=user_input, color=color_mapping['Image Data'])

    elif recommender == 'Audio Data':
        recommendations = recommender_audio(user_input, audio_features_list, audio_track_ids_list, bp_song, top_n=5)
        recommendations['audio_path'] = recommendations['track_id'].apply(
            lambda track_id: f"static/audio_files/{sample_uuid[sample_uuid['track_id'] == track_id]['sample_uuid'].values[0]}.mp3")
        recommendations_list = recommendations[['track_id', 'song', 'audio_path']].to_dict(orient='records')
        return render_template('results_audio.html', input_song=input_song_info, recommendations=recommendations_list, track_id=user_input, color=color_mapping['Audio Data'])

    else:
        error_message = "Invalid recommender selected."
        return render_template('error.html', error_message=error_message)

votes = {'up': 0, 'down': 0}

@app.route('/vote', methods=['POST'])
def vote():
    data = request.get_json()
    vote_type = data.get('vote')

    if vote_type == 'up':
        votes['up'] += 1
        message = "Glad to hear it! Thank you for the feedback!"
    elif vote_type == 'down':
        votes['down'] += 1
        message = "Sorry to hear that! Thank you for the feedback!"
    else:
        message = "Invalid vote."

    return jsonify({'message': message, 'votes': votes})

if __name__ == '__main__':
    app.run(debug=True)
