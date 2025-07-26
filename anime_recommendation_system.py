# Streamlined Anime Recommendation System
# Author: AI Developer | Focused on core recommendation functionality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class AnimeRecommendationSystem:
    """
    Streamlined Anime Recommendation System focused on core functionality.
    
    Features:
    1. Content-based filtering using TF-IDF vectorization
    2. Hybrid recommendation combining content similarity and ratings
    3. Advanced data preprocessing and feature engineering
    4. Intelligent clustering for grouping similar anime
    """
    
    def __init__(self, file_path):
        """Initialize the recommendation system with dataset."""
        self.data = None
        self.processed_data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.scaler = StandardScaler()
        self.kmeans = None
        
        self.load_and_preprocess_data(file_path)
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the anime dataset with enhanced cleaning."""
        print("🔄 Loading and preprocessing dataset...")
        
        try:
            self.data = pd.read_csv(file_path)
            print(f"✅ Dataset loaded: {len(self.data):,} records")
        except FileNotFoundError:
            print("❌ Error: CSV file not found. Please check the file path.")
            return
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return
        
        # Display basic dataset info
        self._display_dataset_info()
        
        # Create processing copy
        self.processed_data = self.data.copy()
        
        # Execute preprocessing pipeline
        self._clean_data()
        self._engineer_features()
        self._handle_outliers()
        
        print(f"✅ Preprocessing completed. Final shape: {self.processed_data.shape}")
    
    def _display_dataset_info(self):
        """Display essential dataset information."""
        print(f"\n📊 Dataset Overview:")
        print(f"   Shape: {self.data.shape}")
        print(f"   Memory: {self.data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show missing values for key columns
        key_columns = ['name', 'genre', 'rating', 'episodes', 'members']
        missing_data = self.data[key_columns].isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n🔍 Missing Values:")
            for col, missing in missing_data[missing_data > 0].items():
                print(f"   {col}: {missing:,} ({missing/len(self.data)*100:.1f}%)")
    
    def _clean_data(self):
        """Enhanced data cleaning with intelligent missing value handling."""
        print("🧹 Cleaning data...")
        
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
        
        print(f"   Cleaned {initial_count - len(self.processed_data):,} invalid records")
    
    def _impute_ratings(self):
        """Intelligent rating imputation using similar anime characteristics."""
        missing_ratings = self.processed_data['rating'].isna()
        
        if missing_ratings.sum() > 0:
            print(f"   Imputing {missing_ratings.sum()} missing ratings...")
            
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
        """Create enhanced features for better recommendations."""
        print("⚙️ Engineering features...")
        
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
        
        print(f"   Created {len(self.processed_data.columns) - len(self.data.columns)} new features")
    
    def _handle_outliers(self):
        """Handle outliers using IQR capping method."""
        print("🎯 Handling outliers...")
        
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
        
        print(f"   Capped {outliers_capped} outlier values")
    
    def build_content_recommender(self):
        """Build content-based recommendation system using TF-IDF."""
        print("🔧 Building content-based recommender...")
        
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
        print("   Computing similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        
        print(f"✅ Content recommender built (Matrix: {self.tfidf_matrix.shape})")
    
    def get_recommendations(self, anime_name, num_recommendations=10):
        """
        Get hybrid recommendations for a given anime.
        
        Args:
            anime_name (str): Name of anime to get recommendations for
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame or str: Recommendations or error message
        """
        try:
            # Find anime with fuzzy matching
            anime_matches = self.processed_data[
                self.processed_data['name'].str.contains(anime_name, case=False, na=False)
            ]
            
            if len(anime_matches) == 0:
                return f"❌ Anime '{anime_name}' not found. Please check the spelling."
            
            # Use the first match
            idx = anime_matches.index[0]
            matched_anime = anime_matches.iloc[0]
            
            print(f"🎯 Found: '{matched_anime['name']}' (Rating: {matched_anime['rating']:.2f})")
            
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
            
            return recommendations
            
        except Exception as e:
            return f"❌ Error generating recommendations: {str(e)}"
    
    def build_clustering_model(self):
        """Build clustering model to group similar anime."""
        print("🔧 Building clustering model...")
        
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
        
        print("   Finding optimal cluster count...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Select optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)
        
        print(f"   Optimal clusters: {optimal_k} (Silhouette: {best_score:.3f})")
        
        # Build final clustering model
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        self.processed_data.loc[X_cluster.index, 'cluster'] = clusters
        
        # Analyze clusters
        self._analyze_clusters(optimal_k)
        
        print(f"✅ Clustering completed with {optimal_k} clusters")
    
    def _analyze_clusters(self, n_clusters):
        """Analyze and describe each cluster."""
        print(f"\n📊 Cluster Analysis:")
        
        for i in range(n_clusters):
            cluster_data = self.processed_data[self.processed_data['cluster'] == i]
            
            if len(cluster_data) == 0:
                continue
                
            print(f"\n🎯 Cluster {i} ({len(cluster_data)} anime):")
            print(f"   Avg Rating: {cluster_data['rating'].mean():.2f}")
            print(f"   Avg Episodes: {cluster_data['episodes'].mean():.1f}")
            print(f"   Most Common Type: {cluster_data['type'].mode().iloc[0] if len(cluster_data['type'].mode()) > 0 else 'Unknown'}")
            
            # Popular genres in cluster
            all_genres = []
            for genres in cluster_data['genre'].dropna():
                all_genres.extend([g.strip() for g in str(genres).split(',')])
            
            if all_genres:
                top_genres = pd.Series(all_genres).value_counts().head(3)
                print(f"   Top Genres: {', '.join(top_genres.index.tolist())}")
    
    def display_system_stats(self):
        """Display comprehensive system statistics."""
        print(f"\n📊 SYSTEM STATISTICS")
        print("=" * 40)
        
        stats = {
            'Total Anime': f"{len(self.processed_data):,}",
            'Average Rating': f"{self.processed_data['rating'].mean():.2f}",
            'Rating Range': f"{self.processed_data['rating'].min():.1f} - {self.processed_data['rating'].max():.1f}",
            'Most Popular Type': self.processed_data['type'].mode().iloc[0],
            'Average Episodes': f"{self.processed_data['episodes'].mean():.1f}",
            'High-Rated Anime (8.0+)': f"{(self.processed_data['rating'] >= 8.0).sum():,}"
        }
        
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Display top genres
        all_genres = []
        for genres in self.processed_data['genre'].dropna():
            all_genres.extend([g.strip() for g in str(genres).split(',')])
        
        top_genres = pd.Series(all_genres).value_counts().head(5)
        print(f"\n🏆 Top 5 Genres:")
        for genre, count in top_genres.items():
            print(f"   {genre}: {count:,}")
    
    def run_interactive_system(self):
        """Launch the interactive recommendation interface."""
        print("\n" + "=" * 60)
        print("    🎌 ANIME RECOMMENDATION SYSTEM 🎌")
        print("=" * 60)
        print("\n🤖 Get personalized anime recommendations powered by AI!")
        print("\n📋 Options:")
        print("   1. 🎯 Get anime recommendations")
        print("   2. 📊 View system statistics") 
        print("   3. 🚪 Exit")
        
        while True:
            try:
                print("\n" + "-" * 40)
                choice = input("🎮 Enter your choice (1-3): ").strip()
                
                if choice == '1':
                    anime_name = input("📝 Enter anime name: ").strip()
                    if not anime_name:
                        print("❌ Please enter a valid anime name.")
                        continue
                        
                    num_recs = input("🔢 Number of recommendations (default 10): ").strip()
                    num_recs = int(num_recs) if num_recs.isdigit() and int(num_recs) > 0 else 10
                    num_recs = min(num_recs, 20)  # Cap at 20
                    
                    print(f"\n🔄 Generating recommendations for '{anime_name}'...")
                    recommendations = self.get_recommendations(anime_name, num_recs)
                    
                    if isinstance(recommendations, pd.DataFrame):
                        print(f"\n✨ Top {len(recommendations)} Recommendations:")
                        print(recommendations.to_string(index=False))
                        
                        # Show summary
                        avg_rating = recommendations['rating'].mean()
                        avg_similarity = recommendations['similarity_score'].mean()
                        print(f"\n📈 Summary:")
                        print(f"   Average Rating: {avg_rating:.2f}")
                        print(f"   Average Similarity: {avg_similarity:.3f}")
                    else:
                        print(recommendations)
                
                elif choice == '2':
                    self.display_system_stats()
                
                elif choice == '3':
                    print("\n🎌 Thank you for using the Anime Recommendation System!")
                    print("✨ Happy anime watching! 👋")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user. Goodbye! 👋")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
                print("🔄 Please try again.")

def main():
    """Main function to initialize and run the recommendation system."""
    print("🚀 ANIME RECOMMENDATION SYSTEM")
    print("🤖 AI-Powered Content Discovery")
    print("=" * 50)
    
    # Initialize system
    system = AnimeRecommendationSystem("anime.csv")
    
    if system.data is None:
        print("❌ Failed to initialize system. Please check your data file.")
        return
    
    try:
        # Build AI models
        print("\n🔧 Building AI models...")
        system.build_content_recommender()
        system.build_clustering_model()
        
        print(f"\n✅ System ready! Built with {len(system.processed_data):,} anime entries.")
        
        # Launch interactive interface
        system.run_interactive_system()
        
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        print("Please check your data file and try again.")

if __name__ == "__main__":
    main()