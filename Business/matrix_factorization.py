import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF, TruncatedSVD

def load(dataset):

    ratings_df = pd.read_csv(dataset)

    # Create a user-item matrix
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')

    # Fill missing values with 0 WARNING HERE
    user_item_matrix = user_item_matrix.fillna(0)

    # Mean centering be aware that it will generate negative values
    meanPoint = user_item_matrix.mean(axis=0)
    user_item_matrix -= meanPoint

    # Convert to a sparse matrix for efficiency
    #user_item_matrix = csr_matrix(user_item_matrix)

    # Set the number of latent factors --- is the k (rank) you can change and tune it
    num_latent_factors = 10

    return user_item_matrix, num_latent_factors, ratings_df


def truncatedSVD(user_item_matrix, num_latent_factors):
    svd = TruncatedSVD(n_components=num_latent_factors)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T

    return user_factors, item_factors

def SVD(user_item_matrix, num_latent_factors):
    #Perform SVD
    u, sigma, vt = svds(user_item_matrix, k=num_latent_factors)

    #Calculate user and item factors
    user_factors = u * sigma
    item_factors = vt.T

    return user_factors, item_factors

def NFM(user_item_matrix, num_latent_factors):
    # Create an NMF model
    model = NMF(n_components=num_latent_factors, max_iter=1000, random_state=42)

    # Fit the model to the user-item matrix
    #user_factors, item_factors = model.fit_transform(user_item_matrix)
    model.fit(user_item_matrix)
    user_factors = model.transform(user_item_matrix)
    item_factors = model.components_.T

    return user_factors, item_factors
def recommend_movies(user_id, user_item_matrix, num_latent_factors, ratings_df):

    user_factors, item_factors = truncatedSVD(user_item_matrix, num_latent_factors)
    #user_factors, item_factors = SVD(user_item_matrix, num_latent_factors)
    #user_factors, item_factors = NFM(user_item_matrix, num_latent_factors)

    user_factor = user_factors[user_id - 1]  # User IDs start from 1 in MovieLens
    item_factor_dot_product = np.dot(item_factors, user_factor)
    top_indices = item_factor_dot_product.argsort()[::-1]
    top_movie_ids = ratings_df['movieId'].unique()[top_indices]
    return top_movie_ids

user_item_matrix, num_latent_factors, ratings_df = load('../Data/ratings.csv')

recommended_movies = recommend_movies(1, user_item_matrix, num_latent_factors, ratings_df)
for movie in recommended_movies[:10]:
    print(movie)