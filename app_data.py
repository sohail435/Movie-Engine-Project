import pandas as pd
import numpy as np

# ==========================================
# 1. DATA DEFINITIONS (Defined first)
# ==========================================

def generate_mock_database():
    """
    Generates a mock relational database using Pandas DataFrames 
    to simulate movie metadata and user profiles.
    """
    movies_data = {
        "movie_id": [101, 102, 103, 104, 105],
        "title": ["The Dark Knight", "Inception", "Interstellar", "The Notebook", "La La Land"],
        "genres": ["Action|Thriller", "Sci-Fi|Thriller", "Sci-Fi|Drama", "Romance|Drama", "Romance|Musical"],
        "description": [
            "A gritty psychological thriller featuring Batman fighting chaos in Gotham.",
            "A thief who steals corporate secrets through the use of dream-sharing technology.",
            "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
            "A poor passionate young man falls in love with a rich young woman.",
            "A jazz pianist and an aspiring actress fall in love while pursuing their dreams in LA."
        ]
    }
    df_movies = pd.DataFrame(movies_data)

    user_profile = {
        "user_id": 1024,
        "watched_movies": [101, 102],  # They watched Dark Knight and Inception
        "preferred_genres": ["Sci-Fi", "Thriller"],
        "historical_ratings": {101: 5, 102: 4}
    }
    
    return df_movies, user_profile


def generate_mock_vector_db(df_movies):
    """
    Simulates a Vector Database by creating numerical embeddings for our movies.
    [Action Weight, Sci-Fi Weight, Romance Weight]
    """
    movie_embeddings = {
        101: np.array([0.9, 0.1, 0.0]),  # The Dark Knight (High Action)
        102: np.array([0.4, 0.9, 0.1]),  # Inception (High Sci-Fi)
        103: np.array([0.1, 0.9, 0.2]),  # Interstellar (Very High Sci-Fi)
        104: np.array([0.0, 0.0, 0.95]), # The Notebook (Pure Romance)
        105: np.array([0.0, 0.2, 0.90])  # La La Land (High Romance)
    }
    return movie_embeddings


# ==========================================
# 2. MATHEMATICAL CORE LOGIC
# ==========================================

def get_user_vector(user_profile):
    """Simulates fetching the user's current behavioral taste profile vector."""
    return np.array([0.6, 0.8, 0.1]) 

def cosine_similarity(v1, v2):
    """Mathematical metric to find vector alignment."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def recommend_movies(user_id, df_movies, user_profile, movie_embeddings, top_n=2):
    user_vec = get_user_vector(user_profile)
    watched = user_profile["watched_movies"]
    
    scores = []
    for _, row in df_movies.iterrows():
        m_id = row["movie_id"]
        
        if m_id in watched:
            continue
            
        movie_vec = movie_embeddings[m_id]
        similarity = cosine_similarity(user_vec, movie_vec)
        
        scores.append({
            "movie_id": m_id,
            "title": row["title"],
            "score": round(float(similarity), 2)
        })
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


# ==========================================
# 3. RUNTIME EXECUTION (Stays at the absolute bottom)
# ==========================================
if __name__ == "__main__":
    df_movies, user_profile = generate_mock_database()
    embeddings = generate_mock_vector_db(df_movies)
    
    print("\n=============================================")
    print("✅ Mock Data Foundations initialized successfully!")
    print(f"Total movies in database: {len(df_movies)}")
    print(f"Target User profile loaded for ID: {user_profile['user_id']}")
    print("=============================================")
    
    recommendations = recommend_movies(1024, df_movies, user_profile, embeddings)
    print("\n🎯 Top Recommendations for User 1024:")
    for rec in recommendations:
        print(f"-> {rec['title']} (Match Score: {rec['score'] * 100}%)")
    print("=============================================\n")