from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import the data foundation and NLP tools we just built!
from app_data import generate_mock_database, generate_mock_vector_db, recommend_movies
from nlp_engine import SentimentAnalyzer

# Initialize our FastAPI Application Gateway
app = FastAPI(
    title="Advanced Movie Recommender & Sentiment Engine API",
    version="1.0.0"
)

# Load our data layers and NLP engine into server memory on startup
df_movies, user_profile = generate_mock_database()
movie_embeddings = generate_mock_vector_db(df_movies)
nlp_analyzer = SentimentAnalyzer()

# Define structural models for API payloads using Pydantic
class ReviewInput(BaseModel):
    user_id: int
    movie_id: int
    review_text: str

@app.get("/")
def home():
    return {"message": "Welcome to the Movie Recommender & Sentiment Analytics API gateway!"}

@app.get("/api/v1/recommendations")
def get_recommendations(user_id: int, top_n: int = 2):
    """
    Flow A: Recommendation Query Path
    Fetches personalized movie predictions via mathematical similarity vectors.
    """
    if user_id != user_profile["user_id"]:
        raise HTTPException(status_code=404, detail="User profile not found in database.")
    
    # Calculate recommendations using our app_data core logic
    results = recommend_movies(user_id, df_movies, user_profile, movie_embeddings, top_n=top_n)
    return {
        "status": "success",
        "user_id": user_id,
        "recommendations": results
    }

@app.post("/api/v1/reviews")
def submit_review(payload: ReviewInput):
    """
    Flow B: Review Submission & Sentiment Feedback Loop Path
    Processes raw text with real Deep Learning and returns structured NLP insights.
    """
    # 1. Run inference through our Hugging Face model pipeline
    nlp_results = nlp_analyzer.analyze_review(payload.review_text)
    
    # 2. Simulate updating the profile vector dynamically based on sentiment feedback
    # In a full database, we would apply this impact multiplier directly to the user vector row!
    status_msg = f"User profile vector shifted by {nlp_results['vector_impact_multiplier']} for this attribute cluster."
    
    return {
        "status": "review_processed",
        "metadata": {
            "user_id": payload.user_id,
            "movie_id": payload.movie_id
        },
        "nlp_analysis": nlp_results,
        "database_feedback_loop": status_msg
    }