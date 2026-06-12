# Netflix Recommendation System

## Project Overview

This project builds a personalized movie recommendation system using the **Netflix Prize Dataset**, implementing and comparing two collaborative filtering approaches — Item-Based CF and SVD Matrix Factorization. Starting from historical user–movie rating data, the system learns user preferences and generates personalised Top-10 content recommendations.

The project covers exploratory data analysis, model design, hyperparameter tuning, evaluation using RMSE and MAP@10, explainable recommendations, and a cold-start fallback strategy.

\---

## Problem Statement

Given historical user–item interaction data from the Netflix Prize Dataset, design and develop a recommendation system capable of learning user preferences, predicting ratings for unseen content, and generating personalised Top-K recommendations. Evaluate and compare at least two recommendation approaches using standard prediction accuracy and ranking quality metrics.

\---

## Models

| Model | Type | Key Parameters |
|-|-|-|
| Item-Based CF | Memory-Based / KNN | k=40, Pearson similarity |
| SVD | Model-Based / Matrix Factorization | n_factors=100, n_epochs=30 |

### How They Work

**Item-Based Collaborative Filtering**

For a given user–movie pair, the model finds the K most similar movies based on shared rating patterns and predicts the rating as a similarity-weighted average:

```
r̂(u, i) = μᵢ + Σ sim(i, j) · (r(u,j) − μⱼ) / Σ |sim(i, j)|
```

**SVD Matrix Factorization**

Decomposes the user–item rating matrix into latent factor matrices, learning hidden user and movie features. The predicted rating is:

```
r̂(u, i) = μ + bᵤ + bᵢ + pᵤ · qᵢᵀ
```

| Symbol | Meaning |
|-|-|
| `μ` | Global mean rating |
| `bᵤ`, `bᵢ` | User and item bias terms |
| `pᵤ`, `qᵢ` | Latent factor vectors for user and item |

\---

## Methodology

1. Parse the Netflix `.txt` rating files (movie IDs on header lines, ratings on subsequent rows)
2. Sample 2 million rows from `combined_data_1.txt` for computational feasibility
3. Filter sparse users (<50 ratings) and movies (<100 ratings) to reduce noise
4. Apply an 80/20 train–test split using scikit-surprise (`random_state=42`)
5. Train Item-Based CF using `KNNWithMeans` with Pearson similarity
6. Tune SVD hyperparameters via 3-fold GridSearch CV, minimising RMSE
7. Evaluate both models using RMSE, MAE, and MAP@10 on the held-out test set
8. Generate Top-10 recommendations for sample users
9. Provide explainable recommendations via Item-CF similarity neighbours
10. Handle cold start with a Bayesian popularity-weighted fallback model

### Relevance Definition

A movie is considered **relevant** for MAP@10 computation if:
```
actual rating ≥ 3.5
```

\---

## Results

| Model | RMSE | MAE | MAP@10 | Train Time |
|-|-|-|-|-|
| Item-Based CF | 0.9720 | 0.7527 | 0.6720 | ~0.1s |
| **SVD** | **0.9273** | **0.7245** | **0.6926** | ~0.4s |

Key observations:

* **SVD outperforms Item-CF** on all quantitative metrics — lower RMSE, lower MAE, and higher MAP@10
* **43% of SVD predictions** fall within 0.5 stars of the true rating
* **Item-Based CF** is more interpretable — recommendations can be explained via similar movies the user has already rated
* **Long-tail movies** with few ratings are harder to predict for both models
* **Cold start** remains unsolved natively — the Bayesian fallback provides a reasonable baseline for new users

SVD best hyperparameters found via GridSearch:

| Hyperparameter | Best Value | Search Range |
|-|-|-|
| n_factors | 100 | {50, 100} |
| n_epochs | 30 | {20, 30} |
| lr_all | 0.005 | {0.005, 0.010} |
| reg_all | 0.02 | {0.02, 0.10} |

\---

## Extensions Explored

* **Explainable Recommendations** — Item-CF similarity neighbours used to generate natural language rationales: *"Recommended because you rated North by Northwest (4.0★) and Cast a Giant Shadow (4.0★) highly"*
* **Cold Start Fallback** — Bayesian-average popularity model for new users with no rating history, balancing average rating and number of ratings to avoid niche over-representation
* Discussion of further extensions: Neural Collaborative Filtering, Two-Tower architecture, Hybrid SVD + content-based filtering, and temporal decay weighting

\---

## Technologies Used

* Python 3
* Pandas
* NumPy
* scikit-surprise
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook / Google Colab

\---

## Repository Structure

```
netflix-recommendation-system/

├── Netflix_Recommendation_System.ipynb

├── Netflix_Recommendation_Technical_Report.pdf

├── Netflix_Recommendation_Presentation.pdf

├── Problem_Statement.pdf

└── README.md
```

\---

## Dataset

The dataset is too large for GitHub and is hosted externally.

| Source | Link |
|-|-|
| Google Drive | https://drive.google.com/drive/folders/1mTqNH0q6Wa4l8JjKiRDr2OWWoEWz97mX?usp=drive_link |
| Kaggle | https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data |

Place the downloaded files inside a folder named `netflix-prize-data` before running the notebook.

\---

## How to Run

Just open `Netflix_Recommendation_System.ipynb` in Google Colab and run the cells sequentially.

The notebook mounts Google Drive automatically. Update the data path in Section 4 if needed:

```python
DATA_DIR = '/content/drive/MyDrive/netflix-prize-data'
```

Download the dataset from Google Drive here:
https://drive.google.com/drive/folders/1mTqNH0q6Wa4l8JjKiRDr2OWWoEWz97mX?usp=drive_link

\---

## References

1. **Surprise Library Documentation**
scikit-surprise is the core library used for collaborative filtering and SVD in this project. Full API reference and algorithm details available here.
[Surprise Docs](https://surprise.readthedocs.io/)

2. **Netflix Prize Dataset**
The original competition dataset released by Netflix in 2006, containing 100M+ ratings from 480K users across 17,770 movies.
[Kaggle Link](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)


\---

## Author

Krish Kumar (                 )
Mohd Saif   (mdsaif1309-coder)

Project submitted as part of the Recommendation Systems for Personalized Content Discovery problem statement.
