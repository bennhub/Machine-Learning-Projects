# Project 1 - Music Mashup Project

## Overview

This project involves extracting, cleaning, and clustering song data from Serato DJ Pro. It starts by parsing raw data from Serato DJ Pro, cleaning the data, and then using a machine learning model to cluster songs based on their musical key and BPM (beats per minute). The end result is a set of clusters that can be used to suggest mashups or create playlists of similar songs.

## Features

- **Data Extraction**: Retrieves and parses raw song data from Serato DJ Pro.
- **Data Cleaning**: Removes special characters and standardizes the data.
- **Feature Extraction**: Extracts and scales features such as musical key and BPM.
- **Clustering**: Uses a neural network model to cluster songs based on key and BPM.
- **Mashup Suggestion**: Finds and suggests songs from the same cluster for potential mashups.

# Song Clustering and Mashup Generator

## Overview

This project is designed to cluster songs based on their musical key and BPM (Beats Per Minute) and find mashup candidates from a given song. The process involves loading song data, preprocessing it, training a neural network for clustering, and then finding songs that are suitable for mashups based on their proximity in the feature space.

## Features

- **Data Loading**: Reads song data from a JSON file.
- **Preprocessing**: Converts musical key notations, imputes missing values, and scales the data.
- **Dimensionality Reduction**: Uses PCA to reduce the number of features.
- **Clustering**: Trains a neural network to cluster songs.
- **Mashup Generation**: Finds songs within the same cluster and ranks them based on similarity to a given song.
- **Output**: Saves the results to a JSON file.

## Prerequisites

- Python 3.6 or higher
- Required Python packages: `json`, `numpy`, `pandas`, `torch`, `scikit-learn`, `scipy`

## Installation

1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```

3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:
    ```bash
    pip install numpy pandas torch scikit-learn scipy
    ```

## Usage

1. **Prepare Data**:
   - Ensure you have a JSON file named `tracks.json` in the `json-files` directory with the song data in the format:
     ```json
     [
       {"track": "Song Title", "key": "C", "bpm": 120},
       ...
     ]
     ```

2. **Update the Script**:
   - Replace the placeholder song title `Got to be enoughtartCon Funk Shun` with an actual song title from your dataset.

3. **Run the Script**:
    ```bash
    python song_mashup_clusterer.py
    ```

4. **Check the Output**:
   - The results, including mashup candidates sorted by distance, will be saved in a JSON file named `playlist.json`.

## Script Details

- **Key Mapping**:
  - Converts Camelot notation of musical keys to integers and vice versa.

- **Data Loading**:
  - Loads song data from a JSON file and converts it to a DataFrame.

- **Preprocessing**:
  - Maps keys to integers, imputes missing values, and normalizes features.
  - Applies PCA for dimensionality reduction.

- **Neural Network for Clustering**:
  - Defines and trains a simple neural network for clustering songs based on key and BPM.

- **Mashup Candidates**:
  - Finds songs in the same cluster as a given song.
  - Calculates Euclidean distances between the given song and candidates.
  - Saves the sorted candidates to a JSON file.

## Example Output

After running the script, the JSON file `playlist.json` will contain the list of songs in the same cluster as the given song, sorted by their distance in the feature space.





# Project 2 - Stock Predictor

## Overview

This project is a stock price predictor that fetches daily stock data from the Alpha Vantage API, processes the data using PyTorch for a linear regression model, and then saves the predicted stock prices to a CSV file. 

## Features

- **Data Fetching**: Retrieves historical stock data from Alpha Vantage.
- **Data Processing**: Cleans and scales the data.
- **Model Training**: Trains a simple linear regression model to predict stock prices.
- **Prediction**: Generates and saves predictions to a CSV file.

## Prerequisites

- Python 3.6 or higher
- Required Python packages: `requests`, `pandas`, `torch`, `scikit-learn`

## Installation

1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```

3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:
    ```bash
    pip install requests pandas torch scikit-learn
    ```

## Usage

1. **Obtain an Alpha Vantage API Key**:
   - Sign up for a free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

2. **Update the Script**:
   - Replace the placeholder API key and stock symbol in `stock_predictor.py` with your actual API key and desired stock symbol.

3. **Run the Script**:
    ```bash
    python stock_predictor.py
    ```

4. **Check the Output**:
   - The predicted stock prices will be saved in `stock_predictions.csv`.

## Script Details

- **`fetch_stock_data(api_key, symbol)`**:
  - Fetches historical stock data from Alpha Vantage and processes it into a DataFrame.

- **Data Scaling**:
  - Scales the stock data features and targets using `StandardScaler`.

- **Model Training**:
  - Defines and trains a simple linear regression model using PyTorch.

- **Prediction**:
  - Generates predictions based on the trained model and saves them to a CSV file.

## Example Output

After running the script, the predictions will be saved in a CSV file named `stock_predictions.csv`, with columns including the original stock data and the predicted close prices.


