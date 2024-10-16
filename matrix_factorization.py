import pandas as pd
import numpy as np
from utils import get_graph

graph = get_graph()

# Load user-movie ratings into DataFrame
ratings_df = pd.DataFrame()

query_ratings = ('MATCH (u:`User`)-[r:`RATED`]->(m:`Movie`) '
                 'RETURN u.id as user_id, m.id as movie_id, '
                 'm.title as movie_title, r.rating as rating LIMIT 2000')

user_movie_ratings = graph.run(query_ratings)
user_id = []
movie_id = []
ratings = []

for record in user_movie_ratings:
    user_id.append(record[0])
    movie_id.append(record[1])
    ratings.append(record[3])

ratings_df['user_id'] = user_id
ratings_df['movie_id'] = movie_id
ratings_df['ratings'] = pd.Series(ratings)
ratings_df['ratings'] = pd.to_numeric(ratings_df['ratings'])

class MF():
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()

        return self

    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

def predict_rating(user_id, movie_id):
    ratings_df_full_pivot = ratings_df.pivot(index='user_id', columns='movie_id', values='ratings').fillna(0)
    user_of_interest_index = ratings_df[(ratings_df['movie_id'] == movie_id) & (ratings_df['user_id'] == user_id)].index
    ratings_df_dr = ratings_df.drop(user_of_interest_index)
    ratings_df_pivot = ratings_df_dr.pivot(index='user_id', columns='movie_id', values='ratings').fillna(0)
    R = np.array(ratings_df_pivot)
    mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=100)
    mf.train()
    return mf.get_rating(user_id, movie_id)
