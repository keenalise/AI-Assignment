import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, silhouette_score)
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎌 Enhanced Anime Recommendation System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAnimeRecommendationSystem:
    """
    Enhanced Anime Recommendation System with Streamlit GUI
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans = None
        self.best_classifier = None
        self.model_scores = {}
    
    @st.cache_data
    def load_and_preprocess_data(_self, file_path=None, uploaded_file=None):
        """Load and preprocess data with caching for better performance"""
        try:
            if uploaded_file is not None:
                _self.data = pd.read_csv(uploaded_file)
            elif file_path:
                _self.data = pd.read_csv(file_path)
            else:
                st.error("No data source provided")
                return False
            
            st.success(f"✅ Dataset loaded successfully: {len(_self.data):,} records")
            
            # Create processed data
            _self.processed_data = _self.data.copy()
            _self._clean_data()
            _self._engineer_features()
            _self._handle_outliers()
            _self._create_target_variables()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def _clean_data(self):
        """Enhanced data cleaning"""
        # Handle missing values
        self.processed_data['genre'] = self.processed_data['genre'].fillna('Unknown')
        self.processed_data['type'] = self.processed_data['type'].fillna('TV')
        
        # Convert episodes to numeric
        self.processed_data['episodes'] = pd.to_numeric(
            self.processed_data['episodes'], errors='coerce'
        )
        
        # Fill missing episodes based on type
        type_episode_median = self.processed_data.groupby('type')['episodes'].median()
        for anime_type, median_episodes in type_episode_median.items():
            mask = (self.processed_data['type'] == anime_type) & \
                   (self.processed_data['episodes'].isna())
            self.processed_data.loc[mask, 'episodes'] = median_episodes
        
        self.processed_data['episodes'] = self.processed_data['episodes'].fillna(1)
        
        # Handle ratings and members
        self.processed_data['rating'] = self.processed_data['rating'].fillna(
            self.processed_data['rating'].median()
        )
        self.processed_data['members'] = self.processed_data['members'].fillna(
            self.processed_data['members'].median()
        )
        
        # Remove duplicates and invalid records
        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data.drop_duplicates(subset=['name'])
        self.processed_data = self.processed_data[self.processed_data['rating'] > 0]
    
    def _engineer_features(self):
        """Advanced feature engineering"""
        # Basic derived features
        self.processed_data['genre_count'] = self.processed_data['genre'].apply(
            lambda x: len(str(x).split(', ')) if pd.notna(x) else 0
        )
        
        # Popularity and engagement metrics
        self.processed_data['popularity_score'] = (
            self.processed_data['members'] * self.processed_data['rating']
        )
        self.processed_data['log_members'] = np.log1p(self.processed_data['members'])
        self.processed_data['log_episodes'] = np.log1p(self.processed_data['episodes'])
        
        # Episode-based features
        self.processed_data['is_movie'] = (self.processed_data['type'] == 'Movie').astype(int)
        self.processed_data['is_long_series'] = (self.processed_data['episodes'] > 24).astype(int)
        self.processed_data['is_short_series'] = (self.processed_data['episodes'] <= 12).astype(int)
        
        # Rating-based features
        self.processed_data['high_rated'] = (self.processed_data['rating'] >= 8.0).astype(int)
        self.processed_data['rating_members_ratio'] = (
            self.processed_data['rating'] / np.log1p(self.processed_data['members'])
        )
        
        # Genre-based features
        popular_genres = ['Comedy', 'Action', 'Drama', 'Romance', 'Fantasy', 'School', 'Supernatural']
        for genre in popular_genres:
            self.processed_data[f'genre_{genre.lower()}'] = self.processed_data['genre'].apply(
                lambda x: 1 if genre in str(x) else 0
            )
    
    def _handle_outliers(self):
        """Handle outliers using IQR method"""
        numerical_columns = ['rating', 'episodes', 'members']
        
        for col in numerical_columns:
            Q1 = self.processed_data[col].quantile(0.25)
            Q3 = self.processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.processed_data[col] = self.processed_data[col].clip(
                lower=max(lower_bound, self.processed_data[col].min()),
                upper=upper_bound
            )
    
    def _create_target_variables(self):
        """Create target variables for classification"""
        self.processed_data['rating_category'] = pd.cut(
            self.processed_data['rating'], 
            bins=[0, 5.5, 6.5, 7.5, 8.5, 10], 
            labels=['Poor', 'Below_Average', 'Average', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        self.processed_data['is_highly_rated'] = (self.processed_data['rating'] >= 8.0).astype(int)
        
        self.processed_data['popularity_category'] = pd.qcut(
            self.processed_data['members'], 
            q=5, 
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        )
    
    def build_content_recommender(self):
        """Build content-based recommendation system"""
        # Create comprehensive feature text
        self.processed_data['enhanced_features'] = (
            self.processed_data['genre'].astype(str) + ' ' + 
            self.processed_data['type'].astype(str) + ' ' +
            pd.cut(self.processed_data['episodes'], 
                  bins=[0, 1, 12, 24, 50, float('inf')], 
                  labels=['Movie', 'Short', 'Standard', 'Long', 'Very_Long']).astype(str)
        )
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.processed_data['enhanced_features']
        )
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, dense_output=False)
    
    def get_recommendations(self, anime_name, num_recommendations=10):
        """Get hybrid recommendations"""
        try:
            # Find anime with fuzzy matching
            anime_matches = self.processed_data[
                self.processed_data['name'].str.contains(anime_name, case=False, na=False)
            ]
            
            if len(anime_matches) == 0:
                return None, f"❌ Anime '{anime_name}' not found"
            
            idx = anime_matches.index[0]
            matched_anime = anime_matches.iloc[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx].toarray()[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get recommendations
            candidate_indices = [i[0] for i in sim_scores[1:num_recommendations*2]]
            candidates = self.processed_data.iloc[candidate_indices].copy()
            
            # Add similarity scores
            candidates['similarity'] = [sim_scores[i+1][1] for i in range(len(candidates))]
            
            # Sort and return top recommendations
            recommendations = candidates.nlargest(num_recommendations, 'similarity')[
                ['name', 'genre', 'rating', 'type', 'episodes', 'members', 'similarity']
            ]
            
            return recommendations, matched_anime
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def build_clustering_model(self):
        """Build clustering model"""
        clustering_features = [
            'rating', 'log_episodes', 'log_members', 'genre_count', 
            'popularity_score', 'is_movie', 'is_long_series', 'high_rated'
        ]
        
        X_cluster = self.processed_data[clustering_features].dropna()
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # Find optimal k using silhouette score
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Build final model
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        self.processed_data.loc[X_cluster.index, 'cluster'] = clusters
        
        return optimal_k, max(silhouette_scores)
    
    def build_classification_model(self):
        """Build classification model"""
        feature_columns = [
            'episodes', 'log_members', 'genre_count', 'popularity_score',
            'is_movie', 'is_long_series', 'is_short_series', 'rating_members_ratio',
            'genre_comedy', 'genre_action', 'genre_drama', 'genre_romance'
        ]
        
        X = self.processed_data[feature_columns].dropna()
        y = self.processed_data.loc[X.index, 'rating_category'].dropna()
        
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest (simplified for demo)
        self.best_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.best_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.best_classifier.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return f1
    
    def predict_rating_category(self, episodes, members, genre_count, anime_type='TV', genres=''):
        """Predict rating category for new anime"""
        if self.best_classifier is None:
            return None
        
        # Engineer features
        log_members = np.log1p(members)
        popularity_score = members * 7.0
        is_movie = 1 if anime_type.lower() == 'movie' else 0
        is_long_series = 1 if episodes > 24 else 0
        is_short_series = 1 if episodes <= 12 else 0
        rating_members_ratio = 7.0 / log_members if log_members > 0 else 0
        
        # Genre features
        genres_lower = genres.lower()
        genre_comedy = 1 if 'comedy' in genres_lower else 0
        genre_action = 1 if 'action' in genres_lower else 0
        genre_drama = 1 if 'drama' in genres_lower else 0
        genre_romance = 1 if 'romance' in genres_lower else 0
        
        # Create feature vector
        features = np.array([[
            episodes, log_members, genre_count, popularity_score,
            is_movie, is_long_series, is_short_series, rating_members_ratio,
            genre_comedy, genre_action, genre_drama, genre_romance
        ]])
        
        # Make prediction
        prediction = self.best_classifier.predict(features)[0]
        probabilities = self.best_classifier.predict_proba(features)[0]
        confidence = max(probabilities)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(self.best_classifier.classes_, probabilities))
        }

# Initialize the system
@st.cache_resource
def get_anime_system():
    return StreamlitAnimeRecommendationSystem()

def main():
    # Header
    st.markdown('<div class="main-header">🎌 Enhanced Anime Recommendation System 🎌</div>', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Powered Anime Discovery Platform")
    
    # Initialize system
    anime_system = get_anime_system()
    
    # Sidebar for file upload and system status
    with st.sidebar:
        st.header("📁 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader("Choose anime dataset CSV", type=['csv'])
        
        # Default file path option
        use_default = st.checkbox("Use default dataset path")
        if use_default:
            default_path = st.text_input("Dataset path:", value="anime_cleaned.csv")
        
        # Load data button
        if st.button("🔄 Load & Process Data"):
            with st.spinner("Loading and processing data..."):
                if uploaded_file is not None:
                    success = anime_system.load_and_preprocess_data(uploaded_file=uploaded_file)
                elif use_default and default_path:
                    success = anime_system.load_and_preprocess_data(file_path=default_path)
                else:
                    st.error("Please upload a file or specify a path")
                    success = False
                
                if success:
                    st.session_state.data_loaded = True
                    st.session_state.models_built = False
    
    # Check if data is loaded
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        st.info("👆 Please load the anime dataset using the sidebar to get started!")
        return
    
    # Dataset overview
    if anime_system.processed_data is not None:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(anime_system.processed_data):,}</h3>
                <p>Total Animes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_rating = anime_system.processed_data['rating'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_rating:.2f}</h3>
                <p>Avg Rating</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_genres = len(set([g.strip() for genres in anime_system.processed_data['genre'].dropna() 
                                   for g in str(genres).split(',') if g.strip()]))
            st.markdown(f"""
            <div class="metric-card">
                <h3>{unique_genres}</h3>
                <p>Unique Genres</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_members = anime_system.processed_data['members'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_members:,.0f}</h3>
                <p>Total Members</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data visualizations
        st.subheader("📈 Data Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig = px.histogram(anime_system.processed_data, x='rating', 
                             title='Rating Distribution', nbins=30)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Type distribution
            type_counts = anime_system.processed_data['type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, 
                        title='Anime Type Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Build models section
    st.header("🔧 AI Model Training")
    
    if st.button("🚀 Build All AI Models"):
        with st.spinner("Building AI models... This may take a moment."):
            progress_bar = st.progress(0)
            
            # Build content recommender
            st.info("Building content-based recommender...")
            anime_system.build_content_recommender()
            progress_bar.progress(33)
            
            # Build clustering model
            st.info("Building clustering model...")
            optimal_k, silhouette = anime_system.build_clustering_model()
            progress_bar.progress(66)
            
            # Build classification model
            st.info("Building classification model...")
            f1_score_val = anime_system.build_classification_model()
            progress_bar.progress(100)
            
            st.session_state.models_built = True
            st.success(f"✅ All models built successfully!")
            st.info(f"🎯 Clustering: {optimal_k} clusters (Silhouette: {silhouette:.3f})")
            st.info(f"📊 Classification F1-Score: {f1_score_val:.3f}")
    
    # Check if models are built
    if not hasattr(st.session_state, 'models_built') or not st.session_state.models_built:
        st.warning("⚠️ Please build the AI models first!")
        return
    
    # Main application tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Get Recommendations", "🔮 Predict Rating", "🔍 Explore Data", "📊 Analytics"])
    
    with tab1:
        st.header("🎯 Anime Recommendations")
        st.write("Get personalized anime recommendations based on your favorite anime!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            anime_name = st.text_input("Enter anime name:", placeholder="e.g., Naruto, Attack on Titan")
        
        with col2:
            num_recs = st.selectbox("Number of recommendations:", [5, 10, 15, 20], index=1)
        
        if st.button("🎌 Get Recommendations", key="rec_button"):
            if anime_name:
                with st.spinner("Finding recommendations..."):
                    recommendations, matched_anime = anime_system.get_recommendations(anime_name, num_recs)
                
                if recommendations is not None:
                    st.success(f"🎯 Found: '{matched_anime['name']}' (Rating: {matched_anime['rating']:.2f})")
                    
                    st.subheader("✨ Recommended Animes")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>🎌 {row['name']}</h4>
                                <p><strong>Genre:</strong> {row['genre']}</p>
                                <p><strong>Rating:</strong> ⭐ {row['rating']:.2f} | 
                                   <strong>Type:</strong> {row['type']} | 
                                   <strong>Episodes:</strong> {row['episodes']} |
                                   <strong>Members:</strong> {row['members']:,}</p>
                                <p><strong>Similarity:</strong> {row['similarity']:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(matched_anime)
            else:
                st.warning("Please enter an anime name!")
    
    with tab2:
        st.header("🔮 Anime Rating Prediction")
        st.write("Predict the rating category for a new anime based on its characteristics!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            episodes = st.number_input("Number of episodes:", min_value=1, max_value=2000, value=12)
            members = st.number_input("Expected members:", min_value=1, max_value=10000000, value=10000)
            genre_count = st.number_input("Number of genres:", min_value=1, max_value=20, value=3)
        
        with col2:
            anime_type = st.selectbox("Anime type:", ["TV", "Movie", "OVA", "Special", "ONA"])
            genres = st.text_input("Genres (comma-separated):", placeholder="e.g., Action, Comedy, Drama")
        
        if st.button("🎯 Predict Rating Category"):
            with st.spinner("Making prediction..."):
                result = anime_system.predict_rating_category(episodes, members, genre_count, anime_type, genres)
            
            if result:
                st.success("🎯 Prediction Results:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Category", result['prediction'])
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col2:
                    # Probability distribution
                    prob_df = pd.DataFrame(list(result['probabilities'].items()), 
                                         columns=['Category', 'Probability'])
                    fig = px.bar(prob_df, x='Category', y='Probability', 
                               title='Category Probabilities')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔍 Data Explorer")
        
        # Search functionality
        st.subheader("🔎 Search Animes")
        search_term = st.text_input("Search for animes:", placeholder="Enter anime name or genre")
        
        if search_term:
            matches = anime_system.processed_data[
                anime_system.processed_data['name'].str.contains(search_term, case=False, na=False)
            ].head(20)
            
            if len(matches) > 0:
                st.write(f"Found {len(matches)} matches:")
                st.dataframe(
                    matches[['name', 'rating', 'type', 'episodes', 'genre', 'members']], 
                    use_container_width=True
                )
            else:
                st.warning("No matches found.")
        
        # Top animes by different criteria
        st.subheader("🏆 Top Animes")
        
        criteria = st.selectbox("Sort by:", ["Rating", "Members", "Episodes"])
        
        if criteria == "Rating":
            top_animes = anime_system.processed_data.nlargest(10, 'rating')
        elif criteria == "Members":
            top_animes = anime_system.processed_data.nlargest(10, 'members')
        else:
            top_animes = anime_system.processed_data.nlargest(10, 'episodes')
        
        st.dataframe(
            top_animes[['name', 'rating', 'type', 'episodes', 'genre', 'members']], 
            use_container_width=True
        )
    
    with tab4:
        st.header("📊 Advanced Analytics")
        
        # Genre analysis
        st.subheader("🎭 Genre Analysis")
        all_genres = []
        for genres in anime_system.processed_data['genre'].dropna():
            all_genres.extend([g.strip() for g in str(genres).split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        
        fig = px.bar(x=genre_counts.index, y=genre_counts.values, 
                    title='Top 10 Most Popular Genres')
        fig.update_layout(xaxis_title='Genre', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating vs Members scatter plot
        st.subheader("📈 Rating vs Members Analysis")
        
        sample_data = anime_system.processed_data.sample(min(1000, len(anime_system.processed_data)))
        fig = px.scatter(sample_data, x='members', y='rating', 
                        color='type', size='episodes',
                        hover_data=['name'], title='Rating vs Members (Sample)')
        fig.update_layout(xaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📋 Summary Statistics")
        summary_stats = anime_system.processed_data[['rating', 'episodes', 'members']].describe()
        st.dataframe(summary_stats, use_container_width=True)

if __name__ == "__main__":
    main()