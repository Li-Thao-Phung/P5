import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent folder to sys.path so Python can find MF_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MF.MF import mf, movie_user_rating, movie_to_idx

# -----------------------------
# Load train and test sets
# -----------------------------
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_path = os.path.join(base_path, "datasets", "ml-32m", "train.csv")
test_path = os.path.join(base_path, "datasets", "ml-32m", "test.csv")

train_ratings = pd.read_csv(train_path)
test_ratings = pd.read_csv(test_path)

# Unique test users
test_users = test_ratings['userId'].unique()





def mmr(user_id, predicted_ratings, movie_embeddings, user_history, lambda_param=0.7, top_k=10):
  relevance_scores = predicted_ratings
  similarity_matrix = cosine_similarity(movie_embeddings)
  selected_indices = []
  # Only include movies the user hasn't already seen
  remaining_indices = [i for i in range(len(relevance_scores)) if not user_history[i]]

  for _ in range(top_k):
    mmr_scores = []
    for i in remaining_indices:
      if selected_indices:
        diversity = max(similarity_matrix[i][j] for j in selected_indices)
      else:
        diversity = 0.0

      mmr_score = lambda_param * relevance_scores[i] - (1 - lambda_param) * diversity
      mmr_scores.append((i,mmr_score))

    best_idx = max(mmr_scores, key=lambda x: x[1])[0]
    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)
  return selected_indices

# -----------------------------
# Parameters
# -----------------------------
lambda_param = 0.7
top_k = 10
movie_embeddings = mf.Q
movie_titles = movie_user_rating.columns.tolist()

all_recommendations = {}

# -----------------------------
# Generate recommendations per test user
# -----------------------------
for user_id in test_users:
    # Predicted ratings from MF
    user_pred_ratings = mf.predict_user_ratings(user_id)

    # Build user history from TRAIN set
    user_history = np.zeros(len(movie_to_idx), dtype=bool)
    seen_movies = train_ratings[train_ratings.userId == user_id]['movieId'].tolist()
    for mid in seen_movies:
        if mid in movie_to_idx:
            user_history[movie_to_idx[mid]] = True

    # Run MMR
    mmr_indices = mmr(
        user_id=user_id,
        predicted_ratings=user_pred_ratings,
        movie_embeddings=movie_embeddings,
        user_history=user_history,
        lambda_param=lambda_param,
        top_k=top_k
    )

    # Map indices to movie titles
    recommended_movies = [movie_titles[i] for i in mmr_indices]
    all_recommendations[user_id] = recommended_movies

    # Print recommendations
    print(f"\nUser {user_id} — Top {top_k} diverse recommendations:")
    for rank, idx in enumerate(mmr_indices, start=1):
        movie = movie_titles[idx]
        rating = user_pred_ratings[idx]
        print(f"{rank}. {movie} — Predicted rating: {rating:.2f}")


# -----------------------------
# Save per-user CSV
# -----------------------------
for user_id, recommended_movies in all_recommendations.items():
    rows = []
    user_pred_ratings = mf.predict_user_ratings(user_id)
    for rank, movie in enumerate(recommended_movies, start=1):
        rating = user_pred_ratings[movie_titles.index(movie)]
        rows.append([user_id, rank, movie, rating])
    df = pd.DataFrame(rows, columns=['user_id', 'rank', 'movie_title', 'predicted_rating'])
    df.to_csv(f"mmr_recommendations_user_{user_id}.csv", index=False)