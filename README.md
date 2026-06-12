🎬 Netflix Recommendation System
Personalized content discovery using Collaborative Filtering and Matrix Factorization on the Netflix Prize Dataset.
---
📊 Results
Model	RMSE	MAE	MAP@10
Item-Based CF	0.9720	0.7527	0.6720
SVD ✅	0.9273	0.7245	0.6926
---
📁 Files in this Repository
File	Description
`Netflix\_Recommendation\_System.ipynb`	Main notebook — full pipeline
`Netflix\_Recommendation\_Technical\_Report.pdf`	10-page technical report
`Netflix\_Recommendation\_Presentation.pdf`	8-slide presentation
`Problem\_Statement.pdf`	Original problem statement
---
🗂️ Dataset
The dataset is too large for GitHub. Download it from one of these links:
Google Drive: https://drive.google.com/drive/folders/1mTqNH0q6Wa4l8JjKiRDr2OWWoEWz97mX?usp=drive_link
Kaggle: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
Once downloaded, place all files inside a folder named `netflix-prize-data`.
---
🚀 How to Run
1. Clone the repository
```bash
git clone https://github.com/your-username/netflix-recommendation-system.git
```
2. Open the notebook in Google Colab
Upload `Netflix\_Recommendation\_System.ipynb` to Google Colab
3. Download the dataset
Get the dataset from Google Drive:
👉 https://drive.google.com/drive/folders/1mTqNH0q6Wa4l8JjKiRDr2OWWoEWz97mX?usp=drive_link
Upload the `netflix-prize-data` folder to your Google Drive, then update this line in Section 4 of the notebook:
```python
DATA\_DIR = '/content/drive/MyDrive/netflix-prize-data'
```
4. Run all cells
```
Runtime → Run all  (Ctrl + F9)
```
---
🔬 Models Used
Item-Based Collaborative Filtering
Finds similar movies based on shared rating patterns
Explainable — can tell users why a movie was recommended
SVD (Matrix Factorization)
Learns hidden user and movie features from ratings
Better accuracy, handles sparse data well
---
💡 Key Findings
SVD achieves better RMSE (0.9273) and MAP@10 (0.6926) than Item-CF
43% of SVD predictions are within 0.5 stars of the true rating
Item-CF is more interpretable despite lower accuracy
Cold start is handled with a Bayesian popularity fallback
---
📚 References
Netflix Prize Dataset — Google Drive
Netflix Prize Dataset — Kaggle
Surprise Library
