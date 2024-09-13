# Song Mashup Clusterer

`song_mashup_clusterer.py` is a Flask-based application that clusters songs based on their musical features (key and BPM) using a neural network. The application allows users to search for mashup candidates by song title, identifying other songs in the same cluster and displaying their proximity based on a PCA-transformed feature space.

## Features

- **Clustering Songs**: Uses a neural network to cluster songs based on their key and BPM.
- **Mashup Candidate Finder**: Enables searching for mashup candidates by song title. Finds songs in the same cluster and ranks them by proximity (distance in feature space).
- **Dimensionality Reduction**: Optionally applies PCA to reduce dimensionality for better clustering performance.
- **Dynamic Mashup Suggestions**: Suggests songs for mashups and ranks them by similarity.

## Requirements

- Python 3.7+
- Flask
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Scipy

Install the required packages by running:

`pip install -r requirements.txt`


## Usage
**Data Preparation:**

Ensure you have a JSON file containing song data at json-files/tracks.json.
The JSON structure should be as follows:

```bash
[
  {
    "track": "Song Title",
    "artist": "Artist Name",
    "key": "C",
    "bpm": 120
  },
  ...
]
```
Start the Flask app: `python song_mashup_clusterer.py`
By default, the app will start in debug mode at **http://127.0.0.1:5000/**

## Clustering Logic
**Neural Network:** A simple feedforward neural network is used to assign clusters to each song. The network is trained on PCA-reduced data (key and BPM).
**Distance Calculation:** Songs in the same cluster are ranked by Euclidean distance in PCA-transformed feature space.
**Example Output**
When searching for mashup candidates for a given song, the app will return a JSON response that looks like this:

```bash
[
  {
    "track": "Candidate Song 1",
    "key": "Em",
    "bpm": 125,
    "cluster": 3,
    "distance": 0.243
  },
  {
    "track": "Candidate Song 2",
    "key": "C",
    "bpm": 120,
    "cluster": 3,
    "distance": 0.354
  }
]
```
## Notes
**Key Mapping:** The key field in the song data is mapped to integers using a custom Camelot-to-integer mapping for processing. After processing, the key is converted back to Camelot notation for display.
**Imputation:** Missing values for key and bpm are imputed using the column mean.
## Customization
You can adjust the neural network architecture, learning rate, or the number of clusters by modifying the corresponding parameters in the script (input_dim, hidden_dim, output_dim, etc.).
PCA can be toggled on/off or adjusted based on your dataset's needs.

