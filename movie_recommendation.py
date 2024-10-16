import pandas as pd
from matrix_factorization import ratings_df

def recommend_movies(user_id, n_recommendations=5):
    user_movies = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()
    
    # Getting movies rated by the user
    recommended_movies = ratings_df[ratings_df['user_id'].isin(user_movies)]
    recommended_movies = recommended_movies.groupby('movie_id').mean().reset_index()
    
    # Return top N recommended movies
    return recommended_movies.sort_values(by='ratings', ascending=False).head(n_recommendations)
