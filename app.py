import streamlit as st
import pandas as pd
from src.recommender import load_data, train_model, get_recommendations

# Page Configuration
st.set_page_config(page_title="Movie Matcher", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie you like, and we'll suggest 5 others!")

# Load Data
@st.cache_data
def get_data_and_model():
    df = load_data('data/tmdb_5000_movies.csv')
    sim_matrix = train_model(df)
    return df, sim_matrix

try:
    movies_df, similarity = get_data_and_model()
    # Dropdown to select movie
    selected_movie = st.selectbox("Type or select a movie:", movies_df['title'].values)

    if st.button('Show Recommendations'):
        recommendations = get_recommendations(selected_movie, movies_df, similarity)
        st.subheader("You might also like:")
        for movie in recommendations:
            st.write(f"⭐ {movie}")

except FileNotFoundError:
    st.error("Error: 'tmdb_5000_credits.csv' not found in 'data/' folder. Please download it.")
