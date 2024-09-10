from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.spatial import distance

app = Flask(__name__)

# Define key mappings from Camelot notation to integers and vice versa
camelot_to_int = {
    'C': 0, 'C#': 1, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 5, 'E': 6, 'F': 7, 'F#': 8, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 12,
    'A': 13, 'A#': 14, 'Bb': 15, 'B': 16, 'Cm': 17, 'C#m': 18, 'Dbm': 19, 'Dm': 20, 'D#m': 21, 'Ebm': 22, 'Em': 23, 'Fm': 24, 'F#m': 25, 'Gbm': 26, 'Gm': 27,
    'G#m': 28, 'Abm': 29, 'Am': 30, 'A#m': 31, 'Bbm': 32, 'Bm': 33
}

int_to_camelot = {v: k for k, v in camelot_to_int.items()}

# Load the JSON file containing song data
with open('json-files/tracks.json', 'r') as file:
    song_data = json.load(file)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(song_data)

# Map the string keys to integers
df['key'] = df['key'].map(camelot_to_int)

# Handle missing values by imputing with the mean for numerical columns
imputer = SimpleImputer(strategy='mean')
df[['key', 'bpm']] = imputer.fit_transform(df[['key', 'bpm']])

# Extract relevant features (key and BPM)
features = df[['key', 'bpm']].values

# Normalize the features for better clustering performance
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction (optional, for better clustering performance)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Convert the data to PyTorch tensors
data = torch.tensor(reduced_features, dtype=torch.float32)

# Define a simple neural network for clustering
class ClusteringNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClusteringNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the parameters
input_dim = 2  # After PCA
hidden_dim = 10
output_dim = 10  # Number of clusters
learning_rate = 0.01
num_epochs = 500

# Initialize the model, loss function, and optimizer
model = ClusteringNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, torch.max(outputs, 1)[1])
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Assign clusters to each song
with torch.no_grad():
    cluster_assignments = torch.argmax(model(data), dim=1).numpy()

df['cluster'] = cluster_assignments

# Function to find songs in the same cluster
def find_mashup_candidates(df, song_title):
    if song_title not in df['track'].values:
        return pd.DataFrame()  # Return an empty DataFrame

    song_cluster = df[df['track'] == song_title]['cluster'].values[0]
    candidates = df[df['cluster'] == song_cluster]
    return candidates

# Replace numerical keys with Camelot notation
df['key'] = df['key'].map(int_to_camelot)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_mashups', methods=['POST'])
def find_mashups():
    song_title = request.json.get('song_title')
    if not song_title:
        return jsonify({"error": "No song title provided"}), 400

    mashup_candidates = find_mashup_candidates(df, song_title)

    if not mashup_candidates.empty:
        song_features = df[df['track'] == song_title][['key', 'bpm']].values
        song_features_int = [camelot_to_int.get(k, np.nan) for k in song_features[:, 0]]  # Ensure numeric conversion
        song_features = np.hstack((np.array(song_features_int).reshape(-1, 1), song_features[:, 1].reshape(-1, 1)))
        
        song_features = scaler.transform(song_features)  # Normalize features
        song_features_pca = pca.transform(song_features).flatten()  # Apply PCA transformation and ensure it's 1-D
        
        def compute_distance(row):
            candidate_features = np.array([camelot_to_int.get(row['key'], np.nan), row['bpm']])
            if np.isnan(candidate_features).any():
                return np.nan
            candidate_features = scaler.transform([candidate_features])
            candidate_features_pca = pca.transform(candidate_features).flatten()  # Ensure it's 1-D
            return distance.euclidean(song_features_pca, candidate_features_pca)
        
        mashup_candidates['distance'] = mashup_candidates.apply(compute_distance, axis=1)
        mashup_candidates = mashup_candidates.sort_values(by='distance')

        results = mashup_candidates[['track', 'key', 'bpm', 'cluster', 'distance']]
        return jsonify(results.to_dict(orient='records'))
    else:
        return jsonify({"message": "No mashup candidates found."})

if __name__ == '__main__':
    app.run(debug=True)
