import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import re
import os

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower().strip()  # Convert to lower case and strip spaces
    return text

def load_and_process_data():
    # Load the movies dataset
    movies_df = pd.read_csv('./archive/rotten_tomatoes_movies.csv')

    # Merge chunks of the reviews dataset
    reviews_chunks_dir = './archive/splits'
    all_chunks = []
    for chunk_file in sorted(os.listdir(reviews_chunks_dir)):
        if chunk_file.endswith('.csv'):
            chunk_path = os.path.join(reviews_chunks_dir, chunk_file)
            chunk_df = pd.read_csv(chunk_path)
            all_chunks.append(chunk_df)
    reviews_df = pd.concat(all_chunks, ignore_index=True)

    # Handle missing titles explicitly
    movies_df['title'] = movies_df['title'].fillna('Unknown Title').str.lower()
    # Exclude entries with 'Unknown Title'
    movies_df = movies_df[movies_df['title'] != 'unknown title']

    # Fill other NaNs with appropriate defaults or placeholders
    movies_df.fillna({'boxOffice': 0, 'audienceScore': 0, 'tomatoMeter': 0}, inplace=True)
    reviews_df.fillna({'reviewText': '', 'scoreSentiment': ''}, inplace=True)

    # Convert text fields to lowercase where applicable and apply preprocessing
    movie_text_cols = ['genre', 'director', 'writer', 'ratingContents', 'originalLanguage']
    review_text_cols = ['reviewText', 'scoreSentiment']

    for col in movie_text_cols:
        if col in movies_df:
            movies_df[col] = movies_df[col].apply(preprocess_text)

    for col in review_text_cols:
        if col in reviews_df:
            reviews_df[col] = reviews_df[col].apply(preprocess_text)

    # Aggregate review data
    reviews_df['processed_reviews'] = reviews_df['reviewText'] + ' ' + reviews_df['scoreSentiment']
    review_summary = reviews_df.groupby('id').agg({
        'processed_reviews': ' '.join,
        'reviewText': 'count'
    }).rename(columns={'reviewText': 'review_count'}).reset_index()

    # Merge reviews into the movies dataset
    movies_df = pd.merge(movies_df, review_summary, on='id', how='left')
    movies_df['processed_reviews'].fillna('', inplace=True)
    movies_df['review_count'].fillna(0, inplace=True)

    # Combine relevant columns into a single feature string for each movie
    feature_columns = [
        'genre', 'director', 'writer', 'ratingContents', 'runtimeMinutes',
        'originalLanguage', 'processed_reviews', 'audienceScore', 'tomatoMeter', 'review_count'
    ]
    movies_df['combined_features'] = movies_df[feature_columns].apply(
        lambda x: ' '.join(x.dropna().astype(str)), axis=1
    )

    return movies_df

def train_knn_model(movies_df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

    # Dimensionality reduction
    svd = TruncatedSVD(n_components=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    tfidf_matrix_reduced = lsa.fit_transform(tfidf_matrix)

    knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(tfidf_matrix_reduced)
    return knn_model, tfidf_vectorizer, lsa

def get_movie_suggestions(title, movies_df, knn_model, tfidf_vectorizer, lsa):
    title = title.lower().strip()
    title_index = movies_df[movies_df['title'].str.lower().str.strip() == title].index
    if title_index.empty:
        return ["Movie title not found."]
    
    title_features = tfidf_vectorizer.transform(movies_df.loc[title_index, 'combined_features'])
    title_features_reduced = lsa.transform(title_features)
    distances, indices = knn_model.kneighbors(title_features_reduced)
    suggestions_indices = [i for i in indices[0] if i != title_index[0]]
    suggestions = movies_df.iloc[suggestions_indices]['title'].tolist()
    return suggestions if suggestions else ["No similar movies found or all are the same as queried movie."]

# Streamlit App
st.title("Movie Recommendation System")

@st.cache_data
def get_data_and_model():
    movies_df = load_and_process_data()
    knn_model, tfidf_vectorizer, lsa = train_knn_model(movies_df)
    return movies_df, knn_model, tfidf_vectorizer, lsa

movies_df, knn_model, tfidf_vectorizer, lsa = get_data_and_model()

movie_title = st.text_input("Enter a movie title for recommendations:")

if movie_title:
    suggestions = get_movie_suggestions(movie_title, movies_df, knn_model, tfidf_vectorizer, lsa)
    st.write(f"Suggestions for '{movie_title}':")
    for suggestion in suggestions:
        st.write(suggestion)