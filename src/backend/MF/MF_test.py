import os
import pandas as pd
import numpy as np

# Dense R 

# -----------------------------
# Paths to dataset
# -----------------------------
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ratings_path = os.path.join(base_path, "datasets", "ml-32m", "ratings.csv")
movies_path = os.path.join(base_path, "datasets", "ml-32m", "movies.csv")


# Load 10,000 ratings for testing
ratings_subset = pd.read_csv(ratings_path, nrows=10000)
movies = pd.read_csv(movies_path)
movies = movies.dropna(subset=['title'])

# Merge ratings and movies
combine_m_r = pd.merge(ratings_subset, movies, on='movieId')
combine_m_r = combine_m_r.drop('timestamp', axis=1)

# -----------------------------
# Create dense user-item matrix
# -----------------------------
movie_user_rating = combine_m_r.pivot(index='userId', columns='title', values='rating').fillna(0)
R = movie_user_rating.values
num_users, num_items = R.shape

# -----------------------------
# Matrix Factorization
# -----------------------------
class MatrixFactorization:
    def __init__(self, R, k=20, alpha=0.01, lamda_=0.1, n_epochs=50):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.alpha = alpha
        self.lambda_ = lamda_
        self.n_epochs = n_epochs

    def train(self):
        # Initialize latent factors and biases
        self.P = np.random.normal(scale=0.1, size=(self.num_users, self.k))
        self.Q = np.random.normal(scale=0.1, size=(self.num_items, self.k))
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.mu = np.mean(self.R[self.R > 0])

        for epoch in range(self.n_epochs):
            for u in range(self.num_users):
                for i in range(self.num_items):
                    if self.R[u, i] > 0:
                        prediction = self.predict_single(u, i)
                        error = self.R[u, i] - prediction

                        # Update parameters
                        self.b_u[u] += self.alpha * (error - self.lambda_ * self.b_u[u])
                        self.b_i[i] += self.alpha * (error - self.lambda_ * self.b_i[i])
                        self.P[u, :] += self.alpha * (error * self.Q[i, :] - self.lambda_ * self.P[u, :])
                        self.Q[i, :] += self.alpha * (error * self.P[u, :] - self.lambda_ * self.Q[i, :])

            loss = self.compute_loss()
            # print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.4f}")

    def predict_single(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)

    def full_prediction(self):
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)

    def compute_loss(self):
        loss = 0
        for u in range(self.num_users):
            for i in range(self.num_items):
                if self.R[u, i] > 0:
                    loss += (self.R[u, i] - self.predict_single(u, i)) ** 2

        # Regularization
        loss += self.lambda_ * (np.sum(self.P**2) + np.sum(self.Q**2) + np.sum(self.b_u**2) + np.sum(self.b_i**2))
        return loss

# -----------------------------
# Train on 10,000 ratings subset
# -----------------------------
mf = MatrixFactorization(R, k=20, alpha=0.01, lamda_=0.1, n_epochs=50)
mf.train()

# -----------------------------
# Generate predictions
# -----------------------------
predicted_ratings = mf.full_prediction()
predicted_ratings_df = pd.DataFrame(predicted_ratings,
                                    index=movie_user_rating.index,
                                    columns=movie_user_rating.columns)

# Top-10 recommendations for user 0
user_idx = 0
user_ratings = predicted_ratings[user_idx, :]
already_rated = movie_user_rating.iloc[user_idx, :] > 0
recommended_idx = [i for i in np.argsort(user_ratings)[::-1] if not already_rated[i]][:10]




print("\n-----------------------------------------------------------------------------")
print("\nTop movies from Matrix Factorization (ranked by predicted rating(MMR)):")
for rank, idx in enumerate(recommended_idx, start=1):
    print(f"{rank}. {movie_user_rating.columns[idx]} — Predicted rating: {user_ratings[idx]:.2f}")


print("\n-----------------------------------------------------------------------------")


__all__ = ['mf', 'movie_user_rating']