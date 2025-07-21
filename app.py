import streamlit as st
import pandas as pd
from anime_recommendation_system import EnhancedAnimeRecommendationSystem

# Load your system
@st.cache_resource
def load_system():
    system = EnhancedAnimeRecommendationSystem("D:\AI assignment\anime_cleaned.csv")  
    # Use the correct path
    system.build_enhanced_content_recommender()
    system.build_optimal_clustering_model()
    system.build_ensemble_classifier()
    return system

st.set_page_config(page_title="Anime Recommender", layout="wide")
st.title("🎌 Enhanced Anime Recommendation System")

system = load_system()

# Sidebar menu
menu = st.sidebar.radio("Choose an option", [
    "🎯 Hybrid Recommendations",
    "🔮 Predict Rating Category",
    "🔍 Search Anime"
])

if menu == "🎯 Hybrid Recommendations":
    st.header("🎯 Get Hybrid Recommendations")
    anime_name = st.text_input("Enter Anime Name")
    num_recs = st.slider("Number of Recommendations", 1, 20, 10)
    
    if st.button("Generate Recommendations"):
        with st.spinner("Finding recommendations..."):
            result = system.get_hybrid_recommendations(anime_name, num_recs)
            if isinstance(result, pd.DataFrame):
                st.success(f"Top {len(result)} Recommendations:")
                st.dataframe(result)
            else:
                st.error(result)

elif menu == "🔮 Predict Rating Category":
    st.header("🔮 Predict Rating Category")
    episodes = st.number_input("Episodes", min_value=1, value=12)
    members = st.number_input("Expected Members", min_value=1, value=1000)
    genre_count = st.number_input("Number of Genres", min_value=1, value=2)
    anime_type = st.selectbox("Type", ["TV", "Movie", "OVA", "Special"])
    genres = st.text_input("Genres (comma-separated)", "Action, Comedy")

    if st.button("Predict"):
        prediction = system.predict_rating_category_enhanced(
            episodes, members, genre_count, anime_type, genres
        )
        if isinstance(prediction, dict):
            st.subheader(f"📊 Predicted Category: {prediction['predicted_category']}")
            st.write(f"🎯 Confidence: {prediction['confidence']:.2%}")
            st.write(f"❓ Uncertainty: {prediction['uncertainty']:.2%}")
            st.info(f"💡 {prediction['explanation']}")
            st.success(f"💭 Recommendation: {prediction['recommendation']}")
            st.write("📋 All Category Probabilities:")
            st.json(prediction['all_probabilities'])
        else:
            st.error(prediction)

elif menu == "🔍 Search Anime":
    st.header("🔍 Search Anime")
    search_term = st.text_input("Search by Name or Keyword")
    if search_term:
        matches = system.processed_data[
            system.processed_data['name'].str.contains(search_term, case=False, na=False)
        ][['name', 'rating', 'type', 'episodes', 'genre']].head(10)
        if len(matches):
            st.dataframe(matches)
        else:
            st.warning("No matches found.")
