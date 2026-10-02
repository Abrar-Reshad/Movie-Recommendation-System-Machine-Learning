# Movie Recommendation System (Machine Learning)

This project implements a multi-model movie recommender using movie metadata such as title, overview, genres, keywords, cast, director, vote average, and vote count.

## Overview

The notebook loads the dataset, preprocesses the text fields, builds TF-IDF features, and then tests several recommendation strategies:

1. Cosine similarity recommender with Nearest Neighbors
2. KNN index over TF-IDF vectors
3. Hybrid similarity using both text and voting features
4. SVM-based genre classification
5. SVM-gated hybrid recommender
6. K-means clustering with cluster-aware recommendations

This makes the project a comparison-based recommendation system rather than a single-model pipeline.

## Implemented methods

### 1) Cosine similarity recommender with Nearest Neighbors
- Builds TF-IDF vectors from movie text fields
- Fits a NearestNeighbors model with cosine distance
- Recommends similar movies for a given title using cosine similarity

### 2) KNN index over TF-IDF vectors
- Uses another KNN index over the same TF-IDF matrix
- Recommends nearest neighbors directly in the TF-IDF feature space

### 3) Hybrid similarity (text + votes)
- Combines TF-IDF text similarity with numeric movie popularity signals
- Uses vote_average and log-transformed vote_count
- Blends both signals using alpha/beta weights

### 4) SVM
- Builds multi-label genre targets from the movie genres
- Trains a One-vs-Rest Linear SVM on TF-IDF features
- Evaluates performance using precision, recall, F1, and accuracy

### 5) SVM-gated hybrid recommender
- Uses the trained SVM to predict likely genres for a query movie
- Filters candidate movies that share predicted genres
- Applies hybrid similarity scoring to the filtered set

### 6) K-means
- Reduces TF-IDF features with LSA (Truncated SVD)
- Runs KMeans clustering on the reduced representation
- Builds cluster-aware recommendation outputs
- Provides cluster summaries and interpretability

## Project structure

- `ML_PROJECT_2.ipynb` – main notebook containing the full workflow
- `README.md` – project overview and usage guide

## Requirements

Install the required Python dependencies:

```bash
pip install pandas numpy scikit-learn pyarrow
```

## Dataset

The notebook expects a CSV file named `movies.csv` in a Google Drive folder such as:

```text
/content/drive/MyDrive/ML Project/movies.csv
```

If your dataset is in a different location, update the `CSV_PATH` variable in the notebook.

## Workflow

1. Load the dataset
2. Normalize column names
3. Validate the movie schema
4. Parse genres into clean list form
5. Build TF-IDF features from multiple text columns
6. Train and test multiple recommendation strategies
7. Evaluate using genre-overlap and NDCG proxy metrics
8. Cluster movies and analyze cluster structure

## Example usage

Open the notebook in Jupyter or VS Code and run each cell in order.

## Recommended environment

- Python 3.9+
- Jupyter Notebook or VS Code with Python extension
- A dataset containing movie metadata including at least:
  - `title`
  - `overview`
  - `genres`
  - `keywords`
  - `cast`
  - `director`
  - `vote_average`
  - `vote_count`

## License

This project is intended for academic and learning use.
