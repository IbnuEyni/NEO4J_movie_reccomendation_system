import streamlit as st
import pandas as pd
from matrix_factorization import predict_rating
from movie_recommendation import recommend_movies

# Title of the app
st.title("Movie Recommendation & Rating App")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ("Predict Rating", "Get Recommendations"))

if app_mode == "Predict Rating":
    st.header("Predict Movie Rating")
    user_id = st.text_input("Enter your user ID:")
    movie_id = st.text_input("Enter the movie ID:")

    if st.button("Predict Rating"):
        if user_id.isdigit() and movie_id.isdigit():
            user_id = int(user_id)
            movie_id = int(movie_id)

            # Predict the rating for the given movie
            predicted_rating = predict_rating(user_id, movie_id)
            st.write(f"Predicted rating for movie ID {movie_id} by user ID {user_id}: {predicted_rating:.2f}")
        else:
            st.error("Please enter valid user ID and movie ID!")

elif app_mode == "Get Recommendations":
    st.header("Get Movie Recommendations")
    user_id = st.text_input("Enter your user ID for recommendations:")

    if st.button("Get Recommendations"):
        if user_id.isdigit():
            user_id = int(user_id)

            # Recommend movies based on the user's rated movies
            recommended_movies = recommend_movies(user_id)
            st.write("Recommended Movies:")
            st.dataframe(recommended_movies)
        else:
            st.error("Please enter a valid user ID!")
