import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(tmdb_5000_movies.csv):
    # Loads and preprocesses the dataset.
    df = pd.read_csv(tmdb_5000_movies.csv)
    df['combined_features'] = df['genres'] + " " + df['overview']
    df['combined_features'] = df['combined_features'].fillna('')
    return df

def train_model(df):
    # Computes the cosine similarity matrix.
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['combined_features'])
    similarity = cosine_similarity(vectors)
    return similarity

def get_recommendations(movie_title, df, similarity_matrix):
    """Returns a list of top 5 recommended movies."""
    try:
        movie_index = df[df['title'] == movie_title].index[0]
        distances = similarity_matrix[movie_index]
        # Sort by similarity score in descending
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
       
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(df.iloc[i[0]].title)
        return recommended_movies
    except IndexError:
        return ["Movie not found in database."]
