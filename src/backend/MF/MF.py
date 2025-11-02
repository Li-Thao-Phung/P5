import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # optional, for progress bars

# Incremnetal R

# -----------------------------
# Paths to dataset
# -----------------------------
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ratings_path = os.path.join(base_path, "datasets", "ml-32m", "ratings.csv")
movies_path = os.path.join(base_path, "datasets", "ml-32m", "movies.csv")

train_path = os.path.join(base_path, "datasets", "ml-32m", "train.csv")
test_path = os.path.join(base_path, "datasets", "ml-32m", "test.csv")

chunksize = 500_000  # number of rows per chunk for incremental reading
test_ratio = 0.2  # 20% test

# Remove old files if they exist
if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(test_path):
    os.remove(test_path)


train_chunks = []
test_chunks = []

for chunk in pd.read_csv(ratings_path, chunksize=chunksize):
    mask = np.random.rand(len(chunk)) < test_ratio
    test_chunk = chunk[mask]
    train_chunk = chunk[~mask]
    
    train_chunk.to_csv(train_path, mode='a', index=False, header=not os.path.exists(train_path))
    test_chunk.to_csv(test_path, mode='a', index=False, header=not os.path.exists(test_path))

# -----------------------------
# Load movies metadata
# -----------------------------
movies = pd.read_csv(movies_path)
movies = movies.dropna(subset=['title'])

# Map movieId to dense 0-based index
movie_ids = movies['movieId'].unique()
movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}


# -----------------------------
# Determine number of users
# -----------------------------
num_users = 0
for chunk in pd.read_csv(ratings_path, usecols=['userId'], chunksize=chunksize):
    num_users = max(num_users, chunk['userId'].max())
num_users += 1  # assuming userId starts from 0 or 1
num_items = len(movie_ids)


class MatrixFactorization:
  def __init__(self, R, k=20, alpha=0.01, lamda_=0.1, n_epochs=50):
    # Now we initialize with the number of users/items, not a dense R
    self.R = R
    self.num_users, self.num_items = R.shape
    self.k = k
    self.alpha = alpha
    self.lambda_ = lamda_
    self.n_epochs = n_epochs

    # Initialize latent factors and biases
    self.P = np.random.normal(scale=0.1, size=(self.num_users, self.k))
    self.Q = np.random.normal(scale=0.1, size=(self.num_items, self.k))
    self.b_u = np.zeros(self.num_users)
    self.b_i = np.zeros(self.num_items)
    self.mu = 0  # will be set based on chunks

  def train(self, ratings_path, movie_to_idx, chunksize=500_000):
        """
        Incrementally trains the model using ratings read in chunks.
        ratings_path : str : path to CSV with columns ['userId', 'movieId', 'rating']
        movie_to_idx : dict : maps movieId to internal 0-based index
        """
        # First pass: compute global mean rating
        total_rating = 0
        count_rating = 0
        for chunk in pd.read_csv(ratings_path, chunksize=chunksize):
            chunk = chunk.dropna(subset=['userId', 'movieId', 'rating'])
            chunk = chunk[chunk['movieId'].isin(movie_to_idx)]
            total_rating += chunk['rating'].sum()
            count_rating += len(chunk)
        self.mu = total_rating / count_rating

        # Training loop over epochs
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            reader = pd.read_csv(ratings_path, chunksize=chunksize)
            for chunk in tqdm(reader, desc="Processing chunks"):
                chunk = chunk.dropna(subset=['userId', 'movieId', 'rating'])
                chunk = chunk[chunk['movieId'].isin(movie_to_idx)]
                chunk['movie_idx'] = chunk['movieId'].map(movie_to_idx)

                # SGD updates for each rating in the chunk
                for row in chunk.itertuples(index=False):
                    u = int(row.userId)
                    i = int(row.movie_idx)
                    r_ui = float(row.rating)
                    pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :])
                    err = r_ui - pred
                    self.b_u[u] += self.alpha * (err - self.lambda_ * self.b_u[u])
                    self.b_i[i] += self.alpha * (err - self.lambda_ * self.b_i[i])
                    self.P[u, :] += self.alpha * (err * self.Q[i, :] - self.lambda_ * self.P[u, :])
                    self.Q[i, :] += self.alpha * (err * self.P[u, :] - self.lambda_ * self.Q[i, :])



  
  def predict_user_ratings(self, user_id):
    return self.mu + self.b_u[user_id] + self.b_i + self.P[user_id, :].dot(self.Q.T)


  def full_prediction(self):
    return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis: , ] + self.P.dot(self.Q.T)


  def compute_loss(self):
    """
    Compute loss over the dataset incrementally if ratings_path is provided
    """
    loss = 0
    if ratings_path:
        reader = pd.read_csv(ratings_path, chunksize=chunksize)
        for chunk in reader:
            chunk = chunk.dropna(subset=['userId', 'movieId', 'rating'])
            chunk = chunk[chunk['movieId'].isin(movie_to_idx)]
            chunk['movie_idx'] = chunk['movieId'].map(movie_to_idx)
            for row in chunk.itertuples(index=False):
                u = int(row.userId)
                i = int(row.movie_idx)
                r_ui = float(row.rating)
                loss += (r_ui - self.predict_single(u, i)) ** 2
    else:
        # fallback to dense R if available
        for u in range(self.num_users):
            for i in range(self.num_items):
                if hasattr(self, 'R') and self.R[u,i] > 0:
                    loss += (self.R[u,i] - self.predict_single(u, i)) ** 2

    # Regularization
    loss += self.lambda_ * (np.sum(self.P**2) + np.sum(self.Q**2) + np.sum(self.b_u**2) + np.sum(self.b_i**2))
    return loss
  


# -----------------------------
# Incremental MF training without creating full dense R
# -----------------------------
# Initialize model with a small placeholder R for shape info
R_placeholder = np.zeros((num_users, len(movie_ids)))
mf = MatrixFactorization(R_placeholder, k=20, alpha=0.01, lamda_=0.1, n_epochs=1)
mf.train(train_path, movie_to_idx, chunksize=10_000)

# -----------------------------
# Generate predictions for MMR
# -----------------------------
user_idx = 0
predicted_ratings = mf.predict_user_ratings(user_idx)
movie_titles = [movies.loc[movies['movieId']==idx_to_movie[i],'title'].values[0] for i in range(len(idx_to_movie))]

num_users = predicted_ratings.shape[0]

for user_idx in range(num_users):
    user_ratings = predicted_ratings[user_idx, :]
    
    # Top-10 recommendations for this user
    recommended_idx = np.argsort(user_ratings)[::-1][:10]
    
    print(f"\nUser {user_idx} — Top 10 recommendations (MF):")
    for rank, idx in enumerate(recommended_idx, start=1):
        print(f"{rank}. {movie_titles[idx]} — Predicted rating: {user_ratings[idx]:.2f}")

__all__ = ['mf', 'movie_user_rating', 'movie_to_idx']