# Movie Recommendation System (Machine Learning)

This project builds a movie recommendation system using a content-based approach with TF-IDF and nearest-neighbor search.

## Overview

The notebook loads a movie dataset, cleans the metadata, parses genres and other text fields, converts those fields into TF-IDF vectors, and then recommends movies using cosine similarity and KNN-based retrieval.

It also includes a simple evaluation step using proxy labels based on shared genres to assess retrieval quality.

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
3. Fix the expected schema
4. Parse genres into clean lists
5. Build TF-IDF features from text metadata
6. Generate recommendations using cosine similarity / KNN
7. Evaluate similarity quality using genre-overlap and NDCG proxy metrics

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

## License

This project is intended for academic and learning use.
