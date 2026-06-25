import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Movie Matcher & Sentiment Engine",
    page_icon="🎬",
    layout="wide"
)

# API endpoints mapping to our FastAPI backend server
BACKEND_URL = "http://127.0.0.1:8000"

st.title("🎬 Advanced Movie Recommender & Sentiment Engine")
st.markdown("---")

# Layout: Split screen into two columns matching our architecture pillars
col1, col2 = st.columns(2)

with col1:
    st.header("🎯 Personalized Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=1024, max_value=1024, value=1024)
    num_rec = st.slider("Number of Recommendations:", min_value=1, max_value=3, value=2)
    
    if st.button("Fetch Recommendations"):
        try:
            # Query Flow A: Hit the FastAPI recommendation gateway
            response = requests.get(
                f"{BACKEND_URL}/api/v1/recommendations", 
                params={"user_id": user_id, "top_n": num_rec}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"Successfully loaded profiles for User {user_id}!")
                
                for idx, rec in enumerate(data["recommendations"]):
                    st.subheader(f"{idx+1}. {rec['title']}")
                    st.progress(int(rec["score"] * 100))
                    st.caption(f"Engine Match Confidence: {rec['score'] * 100}%")
            else:
                st.error("Failed to connect to profile database.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Gateway Connection Error! Is your FastAPI server running on port 8000?")

with col2:
    st.header("🧠 NLP Sentiment Analytics")
    movie_id = st.selectbox("Select Movie to Review:", [101, 102, 103, 104, 105])
    review_text = st.text_area(
        "Write your movie review here:", 
        placeholder="Type something like: 'The visuals were stunning but the plot was messy...'"
    )
    
    if st.button("Analyze Review Sentiment"):
        if review_text.strip() == "":
            st.warning("Please type a review first.")
        else:
            try:
                # Query Flow B: Post textual payload to the NLP backend gateway
                payload = {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    "review_text": review_text
                }
                response = requests.post(f"{BACKEND_URL}/api/v1/reviews", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    nlp = data["nlp_analysis"]
                    
                    st.write("### 📊 Engine Insights:")
                    
                    # Style the output based on deep learning model verdict
                    if nlp["sentiment"] == "POSITIVE":
                        st.success(f"Sentiment: {nlp['sentiment']} (Confidence: {nlp['confidence'] * 100:.2f}%)")
                        st.info(f"🔄 Feedback Loop: {data['database_feedback_loop']}")
                    else:
                        st.error(f"Sentiment: {nlp['sentiment']} (Confidence: {nlp['confidence'] * 100:.2f}%)")
                        st.info(f"🔄 Feedback Loop: {data['database_feedback_loop']}")
                else:
                    st.error("NLP extraction pipeline encountered an error.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Gateway Connection Error! Is your FastAPI server running on port 8000?")