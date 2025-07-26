import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎌 Anime Recommendation System",
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
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .cluster-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #90caf9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAnimeRecommendationSystem:
    """
    Streamlined Anime Recommendation System with Streamlit GUI
    Matches the functionality of the command-line version
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.scaler = StandardScaler()
        self.kmeans = None
    
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
            
            st.success(f"✅ Dataset loaded: {len(_self.data):,} records")
            
            # Display basic dataset info
            _self._display_dataset_info()
            
            # Create processed data
            _self.processed_data = _self.data.copy()
            _self._clean_data()
            _self._engineer_features()
            _self._handle_outliers()
            
            st.success(f"✅ Preprocessing completed. Final shape: {_self.processed_data.shape}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def _display_dataset_info(self):
        """Display essential dataset information"""
        st.info(f"📊 Dataset Overview: Shape {self.data.shape}, Memory: {self.data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show missing values for key columns
        key_columns = ['name', 'genre', 'rating', 'episodes', 'members']
        missing_data = self.data[key_columns].isnull().sum()
        if missing_data.sum() > 0:
            missing_info = []
            for col, missing in missing_data[missing_data > 0].items():
                missing_info.append(f"{col}: {missing:,} ({missing/len(self.data)*100:.1f}%)")
            if missing_info:
                st.warning(f"🔍 Missing Values: {', '.join(missing_info)}")
    
    def _clean_data(self):
        """Enhanced data cleaning with intelligent missing value handling"""
        with st.spinner("🧹 Cleaning data..."):
            # Handle missing values strategically
            self.processed_data['genre'] = self.processed_data['genre'].fillna('Unknown')
            self.processed_data['type'] = self.processed_data['type'].fillna('TV')
            
            # Convert episodes to numeric and handle missing values
            self.processed_data['episodes'] = pd.to_numeric(
                self.processed_data['episodes'], errors='coerce'
            )
            
            # Fill missing episodes based on anime type
            type_medians = self.processed_data.groupby('type')['episodes'].median()
            for anime_type, median_eps in type_medians.items():
                mask = (self.processed_data['type'] == anime_type) & \
                       (self.processed_data['episodes'].isna())
                self.processed_data.loc[mask, 'episodes'] = median_eps
            
            # Handle remaining missing values
            self.processed_data['episodes'] = self.processed_data['episodes'].fillna(1)
            
            # Smart rating imputation
            self._impute_ratings()
            
            # Fill missing members with median
            self.processed_data['members'] = self.processed_data['members'].fillna(
                self.processed_data['members'].median()
            )
            
            # Remove duplicates and invalid records
            initial_count = len(self.processed_data)
            self.processed_data = self.processed_data.drop_duplicates(subset=['name'])
            self.processed_data = self.processed_data[self.processed_data['rating'] > 0]
            
            cleaned_count = initial_count - len(self.processed_data)
            if cleaned_count > 0:
                st.info(f"   Cleaned {cleaned_count:,} invalid records")
    
    def _impute_ratings(self):
        """Intelligent rating imputation using similar anime characteristics"""
        missing_ratings = self.processed_data['rating'].isna()
        
        if missing_ratings.sum() > 0:
            st.info(f"   Imputing {missing_ratings.sum()} missing ratings...")
            
            for idx in self.processed_data[missing_ratings].index:
                # Find similar anime based on genre and type
                current_genre = self.processed_data.loc[idx, 'genre']
                current_type = self.processed_data.loc[idx, 'type']
                
                similar_anime = self.processed_data[
                    (self.processed_data['genre'] == current_genre) &
                    (self.processed_data['type'] == current_type) &
                    (~self.processed_data['rating'].isna())
                ]
                
                if len(similar_anime) > 0:
                    # Weighted average by member count
                    weights = similar_anime['members'] / similar_anime['members'].sum()
                    imputed_rating = (similar_anime['rating'] * weights).sum()
                    self.processed_data.loc[idx, 'rating'] = imputed_rating
                else:
                    # Fallback to overall median
                    self.processed_data.loc[idx, 'rating'] = self.processed_data['rating'].median()
    
    def _engineer_features(self):
        """Create enhanced features for better recommendations"""
        with st.spinner("⚙️ Engineering features..."):
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
            
            # Categorical features
            self.processed_data['is_movie'] = (self.processed_data['type'] == 'Movie').astype(int)
            self.processed_data['is_long_series'] = (self.processed_data['episodes'] > 24).astype(int)
            self.processed_data['high_rated'] = (self.processed_data['rating'] >= 8.0).astype(int)
            
            # Genre indicators for popular genres
            popular_genres = ['Comedy', 'Action', 'Drama', 'Romance', 'Fantasy']
            for genre in popular_genres:
                self.processed_data[f'has_{genre.lower()}'] = self.processed_data['genre'].apply(
                    lambda x: 1 if genre in str(x) else 0
                )
            
            new_features = len(self.processed_data.columns) - len(self.data.columns)
            st.info(f"   Created {new_features} new features")
    
    def _handle_outliers(self):
        """Handle outliers using IQR capping method"""
        with st.spinner("🎯 Handling outliers..."):
            numerical_columns = ['rating', 'episodes', 'members']
            outliers_capped = 0
            
            for col in numerical_columns:
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers before capping
                outliers = ((self.processed_data[col] < lower_bound) | 
                           (self.processed_data[col] > upper_bound)).sum()
                outliers_capped += outliers
                
                # Cap outliers
                self.processed_data[col] = self.processed_data[col].clip(
                    lower=max(lower_bound, self.processed_data[col].min()),
                    upper=upper_bound
                )
            
            if outliers_capped > 0:
                st.info(f"   Capped {outliers_capped} outlier values")
    
    def build_content_recommender(self):
        """Build content-based recommendation system using TF-IDF"""
        with st.spinner("🔧 Building content-based recommender..."):
            # Create comprehensive feature text
            self.processed_data['content_features'] = (
                self.processed_data['genre'].astype(str) + ' ' + 
                self.processed_data['type'].astype(str) + ' ' +
                pd.cut(self.processed_data['episodes'], 
                      bins=[0, 1, 12, 24, 50, float('inf')], 
                      labels=['Movie', 'Short', 'Standard', 'Long', 'Extended']).astype(str)
            )
            
            # Initialize and fit TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.processed_data['content_features']
            )
            
            # Compute cosine similarity matrix
            self.cosine_sim = cosine_similarity(self.tfidf_matrix)
            
            st.success(f"✅ Content recommender built (Matrix: {self.tfidf_matrix.shape})")
    
    def get_recommendations(self, anime_name, num_recommendations=10):
        """Get hybrid recommendations for a given anime"""
        try:
            # Find anime with fuzzy matching
            anime_matches = self.processed_data[
                self.processed_data['name'].str.contains(anime_name, case=False, na=False)
            ]
            
            if len(anime_matches) == 0:
                return None, f"❌ Anime '{anime_name}' not found. Please check the spelling."
            
            # Use the first match
            idx = anime_matches.index[0]
            matched_anime = anime_matches.iloc[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar anime (excluding the input anime)
            similar_indices = [i[0] for i in sim_scores[1:num_recommendations*2]]
            candidates = self.processed_data.iloc[similar_indices].copy()
            
            # Add similarity scores
            candidates['similarity_score'] = [sim_scores[i+1][1] for i in range(len(candidates))]
            
            # Calculate hybrid score (similarity + rating quality)
            candidates['rating_normalized'] = (
                (candidates['rating'] - candidates['rating'].min()) / 
                (candidates['rating'].max() - candidates['rating'].min())
            )
            
            candidates['hybrid_score'] = (
                0.7 * candidates['similarity_score'] + 
                0.3 * candidates['rating_normalized']
            )
            
            # Get final recommendations
            recommendations = candidates.nlargest(num_recommendations, 'hybrid_score')[
                ['name', 'genre', 'rating', 'type', 'episodes', 'members', 'similarity_score']
            ].round({'rating': 2, 'similarity_score': 3})
            
            return recommendations, matched_anime
            
        except Exception as e:
            return None, f"❌ Error generating recommendations: {str(e)}"
    
    def build_clustering_model(self):
        """Build clustering model to group similar anime"""
        with st.spinner("🔧 Building clustering model..."):
            # Select features for clustering
            clustering_features = [
                'rating', 'log_episodes', 'log_members', 'genre_count',
                'popularity_score', 'is_movie', 'is_long_series', 'high_rated'
            ]
            
            X_cluster = self.processed_data[clustering_features].dropna()
            X_scaled = self.scaler.fit_transform(X_cluster)
            
            # Find optimal number of clusters
            silhouette_scores = []
            k_range = range(3, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Select optimal k
            optimal_k = k_range[np.argmax(silhouette_scores)]
            best_score = max(silhouette_scores)
            
            # Build final clustering model
            self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = self.kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to data
            self.processed_data.loc[X_cluster.index, 'cluster'] = clusters
            
            # Store cluster analysis results
            self.cluster_analysis = self._analyze_clusters(optimal_k)
            
            st.success(f"✅ Clustering completed with {optimal_k} clusters (Silhouette: {best_score:.3f})")
            return optimal_k, best_score
    
    def _analyze_clusters(self, n_clusters):
        """Analyze and describe each cluster"""
        cluster_info = {}
        
        for i in range(n_clusters):
            cluster_data = self.processed_data[self.processed_data['cluster'] == i]
            
            if len(cluster_data) == 0:
                continue
            
            # Get top genres in cluster
            all_genres = []
            for genres in cluster_data['genre'].dropna():
                all_genres.extend([g.strip() for g in str(genres).split(',')])
            
            top_genres = []
            if all_genres:
                top_genres = pd.Series(all_genres).value_counts().head(3).index.tolist()
            
            cluster_info[i] = {
                'count': len(cluster_data),
                'avg_rating': cluster_data['rating'].mean(),
                'avg_episodes': cluster_data['episodes'].mean(),
                'most_common_type': cluster_data['type'].mode().iloc[0] if len(cluster_data['type'].mode()) > 0 else 'Unknown',
                'top_genres': top_genres
            }
        
        return cluster_info

# Initialize the system
@st.cache_resource
def get_anime_system():
    return StreamlitAnimeRecommendationSystem()

def main():
    # Header
    st.markdown('<div class="main-header">🎌 Anime Recommendation System 🎌</div>', unsafe_allow_html=True)
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
            default_path = st.text_input("Dataset path:", value="anime.csv")
        
        # Load data button
        if st.button("🔄 Load & Process Data"):
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
            high_rated = (anime_system.processed_data['rating'] >= 8.0).sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{high_rated:,}</h3>
                <p>High-Rated (8.0+)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data visualizations
        st.subheader("📈 Data Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig = px.histogram(anime_system.processed_data, x='rating', 
                             title='Rating Distribution', nbins=30,
                             color_discrete_sequence=['#FF6B6B'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Type distribution
            type_counts = anime_system.processed_data['type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, 
                        title='Anime Type Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Build models section
    st.header("🔧 AI Model Training")
    
    if st.button("🚀 Build All AI Models"):
        progress_bar = st.progress(0)
        
        # Build content recommender
        anime_system.build_content_recommender()
        progress_bar.progress(50)
        
        # Build clustering model
        optimal_k, silhouette = anime_system.build_clustering_model()
        progress_bar.progress(100)
        
        st.session_state.models_built = True
        st.balloons()
        st.success("🎉 All AI models built successfully!")
    
    # Check if models are built
    if not hasattr(st.session_state, 'models_built') or not st.session_state.models_built:
        st.warning("⚠️ Please build the AI models first!")
        return
    
    # Main application tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 System Statistics", "🔍 Explore Data"])
    
    with tab1:
        st.header("🎯 Anime Recommendations")
        st.write("Get personalized anime recommendations based on your favorite anime!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            anime_name = st.text_input("Enter anime name:", placeholder="e.g., Naruto, Attack on Titan, Death Note")
        
        with col2:
            num_recs = st.selectbox("Number of recommendations:", [5, 10, 15, 20], index=1)
        
        if st.button("🎌 Get Recommendations", key="rec_button"):
            if anime_name:
                with st.spinner("🔄 Finding perfect recommendations for you..."):
                    recommendations, matched_anime = anime_system.get_recommendations(anime_name, num_recs)
                
                if recommendations is not None:
                    st.success(f"🎯 Found: '{matched_anime['name']}' (Rating: {matched_anime['rating']:.2f})")
                    
                    st.subheader("✨ Recommended Animes")
                    
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>🎌 {row['name']}</h4>
                            <p><strong>Genre:</strong> {row['genre']}</p>
                            <p><strong>Rating:</strong> ⭐ {row['rating']:.2f} | 
                               <strong>Type:</strong> {row['type']} | 
                               <strong>Episodes:</strong> {int(row['episodes'])} |
                               <strong>Members:</strong> {int(row['members']):,}</p>
                            <p><strong>Similarity Score:</strong> {row['similarity_score']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show summary statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_rating = recommendations['rating'].mean()
                        st.metric("Average Rating", f"{avg_rating:.2f}")
                    with col2:
                        avg_similarity = recommendations['similarity_score'].mean()
                        st.metric("Average Similarity", f"{avg_similarity:.3f}")
                
                else:
                    st.error(matched_anime)
            else:
                st.warning("Please enter an anime name!")
    
    with tab2:
        st.header("📊 System Statistics")
        
        # Overall statistics
        stats = {
            'Total Anime': f"{len(anime_system.processed_data):,}",
            'Average Rating': f"{anime_system.processed_data['rating'].mean():.2f}",
            'Rating Range': f"{anime_system.processed_data['rating'].min():.1f} - {anime_system.processed_data['rating'].max():.1f}",
            'Most Popular Type': anime_system.processed_data['type'].mode().iloc[0],
            'Average Episodes': f"{anime_system.processed_data['episodes'].mean():.1f}",
            'High-Rated Anime (8.0+)': f"{(anime_system.processed_data['rating'] >= 8.0).sum():,}"
        }
        
        st.subheader("📈 Key Metrics")
        for key, value in stats.items():
            st.text(f"{key}: {value}")
        
        # Top genres
        st.subheader("🏆 Top 5 Genres")
        all_genres = []
        for genres in anime_system.processed_data['genre'].dropna():
            all_genres.extend([g.strip() for g in str(genres).split(',')])
        
        top_genres = pd.Series(all_genres).value_counts().head(5)
        
        # Create a bar chart for top genres
        fig = px.bar(x=top_genres.index, y=top_genres.values, 
                    title='Top 5 Most Popular Genres',
                    color_discrete_sequence=['#4ECDC4'])
        fig.update_layout(xaxis_title='Genre', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster analysis if available
        if hasattr(anime_system, 'cluster_analysis'):
            st.subheader("🎪 Cluster Analysis")
            
            for cluster_id, info in anime_system.cluster_analysis.items():
                st.markdown(f"""
                <div class="cluster-card">
                    <h4>🎯 Cluster {cluster_id} ({info['count']} anime)</h4>
                    <p><strong>Average Rating:</strong> {info['avg_rating']:.2f}</p>
                    <p><strong>Average Episodes:</strong> {info['avg_episodes']:.1f}</p>
                    <p><strong>Most Common Type:</strong> {info['most_common_type']}</p>
                    <p><strong>Top Genres:</strong> {', '.join(info['top_genres'])}</p>
                </div>
                """, unsafe_allow_html=True)
    
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
                
                # Display search results in a nice format
                for idx, row in matches.iterrows():
                    with st.expander(f"🎌 {row['name']} - Rating: {row['rating']:.2f}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Type:** {row['type']}")
                            st.write(f"**Episodes:** {int(row['episodes'])}")
                            st.write(f"**Members:** {int(row['members']):,}")
                        with col2:
                            st.write(f"**Genre:** {row['genre']}")
                            if 'cluster' in row:
                                st.write(f"**Cluster:** {int(row['cluster']) if pd.notna(row['cluster']) else 'N/A'}")
            else:
                st.warning("No matches found. Try a different search term.")
        
        # Top animes by different criteria
        st.subheader("🏆 Top Animes")
        
        criteria = st.selectbox("Sort by:", ["Rating", "Members", "Episodes"])
        
        if criteria == "Rating":
            top_animes = anime_system.processed_data.nlargest(10, 'rating')
            st.write("Top 10 Highest Rated Animes:")
        elif criteria == "Members":
            top_animes = anime_system.processed_data.nlargest(10, 'members')
            st.write("Top 10 Most Popular Animes (by Members):")
        else:
            top_animes = anime_system.processed_data.nlargest(10, 'episodes')
            st.write("Top 10 Longest Animes (by Episodes):")
        
        # Display top animes in cards
        for idx, row in top_animes.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>🎌 {row['name']}</h4>
                    <p><strong>Genre:</strong> {row['genre']}</p>
                    <p><strong>Rating:</strong> ⭐ {row['rating']:.2f} | 
                       <strong>Type:</strong> {row['type']} | 
                       <strong>Episodes:</strong> {int(row['episodes'])} |
                       <strong>Members:</strong> {int(row['members']):,}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Advanced analytics
        st.subheader("📈 Advanced Analytics")
        
        # Rating vs Members scatter plot
        st.write("**Rating vs Members Relationship**")
        sample_size = min(1000, len(anime_system.processed_data))
        sample_data = anime_system.processed_data.sample(sample_size)
        
        fig = px.scatter(sample_data, x='members', y='rating', 
                        color='type', size='episodes',
                        hover_data=['name'], 
                        title=f'Rating vs Members Analysis (Sample of {sample_size})',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(xaxis_type="log", xaxis_title="Members (log scale)", yaxis_title="Rating")
        st.plotly_chart(fig, use_container_width=True)
        
        # Episodes distribution by type
        st.write("**Episodes Distribution by Type**")
        fig = px.box(anime_system.processed_data, x='type', y='episodes', 
                    title='Episodes Distribution by Anime Type',
                    color_discrete_sequence=['#FF6B6B'])
        fig.update_layout(yaxis_type="log", yaxis_title="Episodes (log scale)")
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎌 <strong>Anime Recommendation System</strong> - Powered by AI & Machine Learning</p>
        <p>Built with Streamlit, scikit-learn, and lots of ❤️ for anime!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()