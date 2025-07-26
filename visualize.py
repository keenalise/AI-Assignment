# Simplified Anime Recommendation System Visualizer
# Author: AI Developer (10+ years experience) | Focused on Core Functionality

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

# Add the path to import the recommendation system
sys.path.append(r'D:\AI assignment')

# Try to import the recommendation system
try:
    from anime_recommendation_system import AnimeRecommendationSystem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import recommendation system: {e}")
    SYSTEM_AVAILABLE = False

warnings.filterwarnings('ignore')

class SimpleAnimeVisualizer:
    """
    Simplified visualization engine for anime recommendation system analysis.
    Uses only matplotlib and seaborn for maximum compatibility.
    """
    
    def __init__(self, csv_path="anime.csv"):
        """Initialize the visualization engine."""
        self.csv_path = csv_path
        self.recommendation_system = None
        self.data = None
        self.processed_data = None
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        self.setup_system()
    
    def setup_system(self):
        """Initialize the system and load data."""
        print("🚀 Initializing Simple Anime Visualizer...")
        print("=" * 50)
        
        try:
            if SYSTEM_AVAILABLE:
                # Initialize recommendation system
                self.recommendation_system = AnimeRecommendationSystem(self.csv_path)
                
                if self.recommendation_system.data is None:
                    raise Exception("Failed to load recommendation system")
                
                self.data = self.recommendation_system.data.copy()
                self.processed_data = self.recommendation_system.processed_data.copy()
                
                # Build models
                print("🔧 Building models...")
                self.recommendation_system.build_content_recommender()
                self.recommendation_system.build_clustering_model()
                
            else:
                # Load data directly if recommendation system not available
                print("📊 Loading data directly...")
                self.data = pd.read_csv(self.csv_path)
                self.processed_data = self._basic_preprocessing(self.data.copy())
            
            print(f"✅ System initialized with {len(self.data):,} anime entries")
            
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            raise
    
    def _basic_preprocessing(self, data):
        """Basic preprocessing if recommendation system not available."""
        print("🧹 Basic data preprocessing...")
        
        # Handle missing values
        data['genre'] = data['genre'].fillna('Unknown')
        data['type'] = data['type'].fillna('TV')
        data['episodes'] = pd.to_numeric(data['episodes'], errors='coerce').fillna(1)
        data['rating'] = pd.to_numeric(data['rating'], errors='coerce').fillna(data['rating'].median())
        data['members'] = pd.to_numeric(data['members'], errors='coerce').fillna(data['members'].median())
        
        # Remove invalid records
        data = data[data['rating'] > 0]
        data = data.drop_duplicates(subset=['name'])
        
        return data
    
    def create_basic_overview(self):
        """Create basic dataset overview."""
        print("\n📊 Creating basic dataset overview...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎌 Anime Dataset Overview', fontsize=16, fontweight='bold')
        
        # Rating distribution
        axes[0, 0].hist(self.processed_data['rating'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episodes distribution (log scale)
        episodes_log = np.log1p(self.processed_data['episodes'])
        axes[0, 1].hist(episodes_log, bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Episodes Distribution (Log Scale)')
        axes[0, 1].set_xlabel('Log(Episodes + 1)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Type distribution
        type_counts = self.processed_data['type'].value_counts().head(10)
        axes[0, 2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Anime Type Distribution')
        
        # Members distribution (log scale)
        members_log = np.log1p(self.processed_data['members'])
        axes[1, 0].hist(members_log, bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Members Distribution (Log Scale)')
        axes[1, 0].set_xlabel('Log(Members + 1)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rating vs Members scatter (sample)
        sample_data = self.processed_data.sample(min(1000, len(self.processed_data)))
        axes[1, 1].scatter(sample_data['rating'], np.log1p(sample_data['members']), 
                          alpha=0.6, color='purple', s=20)
        axes[1, 1].set_title('Rating vs Members (Log Scale)')
        axes[1, 1].set_xlabel('Rating')
        axes[1, 1].set_ylabel('Log(Members + 1)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Top genres
        all_genres = []
        for genres in self.processed_data['genre'].dropna():
            all_genres.extend([g.strip() for g in str(genres).split(',')])
        
        top_genres = pd.Series(all_genres).value_counts().head(10)
        axes[1, 2].barh(range(len(top_genres)), top_genres.values, color='orange', alpha=0.7)
        axes[1, 2].set_yticks(range(len(top_genres)))
        axes[1, 2].set_yticklabels(top_genres.index)
        axes[1, 2].set_title('Top 10 Genres')
        axes[1, 2].set_xlabel('Count')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('anime_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Basic overview saved as 'anime_overview.png'")
    
    def create_genre_analysis(self):
        """Create genre analysis visualizations."""
        print("\n🎭 Creating genre analysis...")
        
        # Extract genres and calculate statistics
        all_genres = []
        genre_ratings = {}
        genre_members = {}
        
        for idx, row in self.processed_data.iterrows():
            if pd.notna(row['genre']):
                genres = [g.strip() for g in str(row['genre']).split(',')]
                all_genres.extend(genres)
                
                for genre in genres:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                        genre_members[genre] = []
                    
                    genre_ratings[genre].append(row['rating'])
                    genre_members[genre].append(row['members'])
        
        # Calculate genre statistics
        genre_stats = {}
        for genre in set(all_genres):
            if len(genre_ratings[genre]) >= 5:  # Minimum 5 anime per genre
                genre_stats[genre] = {
                    'count': len(genre_ratings[genre]),
                    'avg_rating': np.mean(genre_ratings[genre]),
                    'avg_members': np.mean(genre_members[genre])
                }
        
        genre_df = pd.DataFrame(genre_stats).T.sort_values('count', ascending=False)
        top_genres = genre_df.head(15)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎭 Genre Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Genre popularity
        axes[0, 0].bar(range(len(top_genres)), top_genres['count'], color='skyblue', alpha=0.7)
        axes[0, 0].set_xticks(range(len(top_genres)))
        axes[0, 0].set_xticklabels(top_genres.index, rotation=45, ha='right')
        axes[0, 0].set_title('Genre Popularity (Count)')
        axes[0, 0].set_ylabel('Number of Anime')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average rating by genre
        axes[0, 1].bar(range(len(top_genres)), top_genres['avg_rating'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_xticks(range(len(top_genres)))
        axes[0, 1].set_xticklabels(top_genres.index, rotation=45, ha='right')
        axes[0, 1].set_title('Average Rating by Genre')
        axes[0, 1].set_ylabel('Average Rating')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average members by genre
        axes[1, 0].bar(range(len(top_genres)), top_genres['avg_members'], color='lightcoral', alpha=0.7)
        axes[1, 0].set_xticks(range(len(top_genres)))
        axes[1, 0].set_xticklabels(top_genres.index, rotation=45, ha='right')
        axes[1, 0].set_title('Average Members by Genre')
        axes[1, 0].set_ylabel('Average Members')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rating vs popularity scatter
        axes[1, 1].scatter(top_genres['avg_rating'], top_genres['avg_members'], 
                          s=top_genres['count']*2, alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Average Rating')
        axes[1, 1].set_ylabel('Average Members')
        axes[1, 1].set_title('Rating vs Popularity (Size = Count)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add genre labels
        for i, genre in enumerate(top_genres.index):
            if i < 5:  # Only label top 5 to avoid clutter
                axes[1, 1].annotate(genre, 
                                  (top_genres.loc[genre, 'avg_rating'], 
                                   top_genres.loc[genre, 'avg_members']),
                                  xytext=(5, 5), textcoords='offset points', 
                                  fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Genre analysis saved as 'genre_analysis.png'")
    
    def create_rating_analysis(self):
        """Create rating distribution analysis."""
        print("\n📈 Creating rating analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 Rating Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Rating distribution by type
        anime_types = self.processed_data['type'].value_counts().head(5).index
        for i, anime_type in enumerate(anime_types):
            type_data = self.processed_data[self.processed_data['type'] == anime_type]
            axes[0, 0].hist(type_data['rating'], bins=15, alpha=0.6, 
                           label=anime_type, density=True)
        
        axes[0, 0].set_title('Rating Distribution by Type')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episodes vs Rating
        sample_data = self.processed_data.sample(min(1000, len(self.processed_data)))
        scatter = axes[0, 1].scatter(sample_data['episodes'], sample_data['rating'], 
                                   c=np.log1p(sample_data['members']), 
                                   alpha=0.6, s=20, cmap='viridis')
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Rating')
        axes[0, 1].set_title('Episodes vs Rating (Color = Log Members)')
        plt.colorbar(scatter, ax=axes[0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot by type
        type_ratings = []
        type_labels = []
        for anime_type in anime_types:
            type_data = self.processed_data[self.processed_data['type'] == anime_type]
            type_ratings.append(type_data['rating'].values)
            type_labels.append(anime_type)
        
        axes[1, 0].boxplot(type_ratings, labels=type_labels)
        axes[1, 0].set_title('Rating Distribution by Type (Box Plot)')
        axes[1, 0].set_ylabel('Rating')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # High-rated anime analysis
        rating_bins = [0, 6, 7, 8, 8.5, 10]
        rating_labels = ['Poor\n(0-6)', 'Average\n(6-7)', 'Good\n(7-8)', 'Great\n(8-8.5)', 'Excellent\n(8.5-10)']
        self.processed_data['rating_category'] = pd.cut(self.processed_data['rating'], 
                                                       bins=rating_bins, labels=rating_labels)
        
        rating_counts = self.processed_data['rating_category'].value_counts()
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        axes[1, 1].bar(rating_counts.index, rating_counts.values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Rating Categories Distribution')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rating_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Rating analysis saved as 'rating_analysis.png'")
    
    def create_recommendation_test(self):
        """Test recommendation system if available."""
        if not SYSTEM_AVAILABLE or self.recommendation_system is None:
            print("⚠️ Recommendation system not available for testing")
            return
        
        print("\n🎯 Testing recommendation system...")
        
        # Test popular anime
        test_anime = ["Naruto", "Death Note", "Attack on Titan", "One Piece"]
        results = []
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎯 Recommendation System Test Results', fontsize=16, fontweight='bold')
        
        for i, anime in enumerate(test_anime):
            try:
                recs = self.recommendation_system.get_recommendations(anime, 10)
                if isinstance(recs, pd.DataFrame) and len(recs) > 0:
                    results.append({
                        'anime': anime,
                        'avg_rating': recs['rating'].mean(),
                        'avg_similarity': recs['similarity_score'].mean(),
                        'count': len(recs)
                    })
                    
                    # Plot recommendations for first 4 anime
                    if i < 4:
                        row, col = i // 2, i % 2
                        axes[row, col].bar(range(len(recs)), recs['rating'], alpha=0.7)
                        axes[row, col].set_title(f'Recommendations for {anime}')
                        axes[row, col].set_ylabel('Rating')
                        axes[row, col].set_xlabel('Recommendation Rank')
                        axes[row, col].grid(True, alpha=0.3)
                        
                        # Add average line
                        axes[row, col].axhline(y=recs['rating'].mean(), color='red', 
                                             linestyle='--', alpha=0.8, 
                                             label=f'Avg: {recs["rating"].mean():.2f}')
                        axes[row, col].legend()
                        
            except Exception as e:
                print(f"   ⚠️ Could not test {anime}: {str(e)}")
        
        plt.tight_layout()
        plt.savefig('recommendation_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary
        if results:
            results_df = pd.DataFrame(results)
            print(f"\n📊 Recommendation Test Summary:")
            print(f"   Average Recommendation Rating: {results_df['avg_rating'].mean():.2f}")
            print(f"   Average Similarity Score: {results_df['avg_similarity'].mean():.3f}")
        
        print("✅ Recommendation test saved as 'recommendation_test.png'")
    
    def create_clustering_analysis(self):
        """Create clustering analysis if available."""
        if not SYSTEM_AVAILABLE or self.recommendation_system is None:
            print("⚠️ Clustering analysis requires recommendation system")
            return
        
        # Try to build clustering if not available
        if 'cluster' not in self.processed_data.columns:
            print("🔧 Building clustering model...")
            try:
                self.recommendation_system.build_clustering_model()
                # Update processed data
                self.processed_data = self.recommendation_system.processed_data.copy()
                print("✅ Clustering model built successfully!")
            except Exception as e:
                print(f"❌ Failed to build clustering model: {str(e)}")
                print("⚠️ Creating basic clustering analysis instead...")
                self._create_basic_clustering()
                return
        
        if 'cluster' not in self.processed_data.columns:
            print("⚠️ No clustering data available")
            return
        
        print("\n🔬 Creating clustering analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔬 Clustering Analysis', fontsize=16, fontweight='bold')
        
        # Cluster distribution
        cluster_counts = self.processed_data['cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Anime')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rating by cluster
        cluster_ratings = []
        cluster_labels = []
        for cluster in sorted(self.processed_data['cluster'].unique()):
            cluster_data = self.processed_data[self.processed_data['cluster'] == cluster]
            cluster_ratings.append(cluster_data['rating'].values)
            cluster_labels.append(f'Cluster {cluster}')
        
        axes[0, 1].boxplot(cluster_ratings, labels=cluster_labels)
        axes[0, 1].set_title('Rating Distribution by Cluster')
        axes[0, 1].set_ylabel('Rating')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episodes by cluster
        cluster_episodes = []
        for cluster in sorted(self.processed_data['cluster'].unique()):
            cluster_data = self.processed_data[self.processed_data['cluster'] == cluster]
            cluster_episodes.append(cluster_data['episodes'].values)
        
        axes[1, 0].boxplot(cluster_episodes, labels=cluster_labels)
        axes[1, 0].set_title('Episodes Distribution by Cluster')
        axes[1, 0].set_ylabel('Episodes')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster characteristics heatmap
        cluster_stats = self.processed_data.groupby('cluster')[['rating', 'episodes', 'members']].mean()
        
        # Simple heatmap using imshow
        im = axes[1, 1].imshow(cluster_stats.T, cmap='viridis', aspect='auto')
        axes[1, 1].set_xticks(range(len(cluster_stats)))
        axes[1, 1].set_xticklabels([f'Cluster {i}' for i in cluster_stats.index])
        axes[1, 1].set_yticks(range(len(cluster_stats.columns)))
        axes[1, 1].set_yticklabels(cluster_stats.columns)
        axes[1, 1].set_title('Cluster Characteristics Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Clustering analysis saved as 'clustering_analysis.png'")
    
    def _create_basic_clustering(self):
        """Create basic clustering using simple K-means if advanced clustering fails."""
        print("🔧 Creating basic clustering analysis...")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Select features for basic clustering
            features = ['rating', 'episodes', 'members']
            
            # Prepare data
            cluster_data = self.processed_data[features].copy()
            cluster_data = cluster_data.dropna()
            
            if len(cluster_data) == 0:
                print("❌ No valid data for clustering")
                return
            
            # Handle extreme values
            for col in features:
                Q1 = cluster_data[col].quantile(0.25)
                Q3 = cluster_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cluster_data[col] = cluster_data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Apply K-means clustering
            n_clusters = min(5, len(cluster_data) // 50)  # Reasonable number of clusters
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add clusters to data
            self.processed_data.loc[cluster_data.index, 'basic_cluster'] = clusters
            
            print(f"✅ Basic clustering completed with {n_clusters} clusters")
            
            # Create basic clustering visualization
            self._visualize_basic_clusters(features, 'basic_cluster')
            
        except ImportError:
            print("❌ scikit-learn not available for clustering")
            print("💡 Install with: pip install scikit-learn")
        except Exception as e:
            print(f"❌ Basic clustering failed: {str(e)}")
    
    def _visualize_basic_clusters(self, features, cluster_column):
        """Visualize basic clustering results."""
        if cluster_column not in self.processed_data.columns:
            print("❌ No cluster data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔬 Basic Clustering Analysis', fontsize=16, fontweight='bold')
        
        # Cluster distribution
        cluster_counts = self.processed_data[cluster_column].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Anime')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rating by cluster
        cluster_data = self.processed_data.dropna(subset=[cluster_column])
        cluster_ratings = []
        cluster_labels = []
        for cluster in sorted(cluster_data[cluster_column].unique()):
            data = cluster_data[cluster_data[cluster_column] == cluster]
            cluster_ratings.append(data['rating'].values)
            cluster_labels.append(f'Cluster {int(cluster)}')
        
        axes[0, 1].boxplot(cluster_ratings, labels=cluster_labels)
        axes[0, 1].set_title('Rating Distribution by Cluster')
        axes[0, 1].set_ylabel('Rating')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot: Rating vs Episodes colored by cluster
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_data[cluster_column].unique())))
        for i, cluster in enumerate(sorted(cluster_data[cluster_column].unique())):
            data = cluster_data[cluster_data[cluster_column] == cluster]
            axes[1, 0].scatter(data['rating'], data['episodes'], 
                              c=[colors[i]], label=f'Cluster {int(cluster)}', 
                              alpha=0.6, s=20)
        
        axes[1, 0].set_xlabel('Rating')
        axes[1, 0].set_ylabel('Episodes')
        axes[1, 0].set_title('Rating vs Episodes by Cluster')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster characteristics
        cluster_stats = cluster_data.groupby(cluster_column)[['rating', 'episodes', 'members']].mean()
        
        x = np.arange(len(cluster_stats))
        width = 0.25
        
        axes[1, 1].bar(x - width, cluster_stats['rating'], width, label='Rating', alpha=0.7)
        axes[1, 1].bar(x, cluster_stats['episodes']/10, width, label='Episodes/10', alpha=0.7)  # Scale down episodes
        axes[1, 1].bar(x + width, cluster_stats['members']/10000, width, label='Members/10k', alpha=0.7)  # Scale down members
        
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Scaled Values')
        axes[1, 1].set_title('Cluster Characteristics (Scaled)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'Cluster {int(i)}' for i in cluster_stats.index])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('basic_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print cluster summary
        print(f"\n🎯 Basic Cluster Summary:")
        print("=" * 40)
        for cluster in sorted(cluster_data[cluster_column].unique()):
            data = cluster_data[cluster_data[cluster_column] == cluster]
            print(f"\n🔸 Cluster {int(cluster)} ({len(data)} anime):")
            print(f"   Average Rating: {data['rating'].mean():.2f}")
            print(f"   Average Episodes: {data['episodes'].mean():.1f}")
            print(f"   Average Members: {data['members'].mean():,.0f}")
        
        print("✅ Basic clustering analysis saved as 'basic_clustering_analysis.png'")
    
    def generate_summary_report(self):
        """Generate a comprehensive text report."""
        print("\n📋 Generating summary report...")
        
        report = []
        report.append("=" * 60)
        report.append("🎌 ANIME RECOMMENDATION SYSTEM - ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset summary
        report.append("📊 DATASET SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Anime Entries: {len(self.data):,}")
        report.append(f"After Processing: {len(self.processed_data):,}")
        report.append(f"Average Rating: {self.processed_data['rating'].mean():.2f}")
        report.append(f"Rating Range: {self.processed_data['rating'].min():.1f} - {self.processed_data['rating'].max():.1f}")
        report.append(f"Total Members: {self.processed_data['members'].sum():,}")
        report.append("")
        
        # Type distribution
        report.append("🎭 ANIME TYPE DISTRIBUTION")
        report.append("-" * 30)
        type_dist = self.processed_data['type'].value_counts()
        for anime_type, count in type_dist.head(8).items():
            percentage = (count / len(self.processed_data)) * 100
            report.append(f"{anime_type:12}: {count:6,} ({percentage:4.1f}%)")
        report.append("")
        
        # Genre analysis
        report.append("🎨 TOP GENRES")
        report.append("-" * 30)
        all_genres = []
        for genres in self.processed_data['genre'].dropna():
            all_genres.extend([g.strip() for g in str(genres).split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts()
        for genre, count in genre_counts.head(10).items():
            percentage = (count / len(self.processed_data)) * 100
            report.append(f"{genre:15}: {count:5,} ({percentage:4.1f}%)")
        report.append("")
        
        # Quality analysis
        report.append("⭐ QUALITY ANALYSIS")
        report.append("-" * 30)
        high_rated = self.processed_data[self.processed_data['rating'] >= 8.0]
        report.append(f"High-rated anime (8.0+): {len(high_rated):,} ({len(high_rated)/len(self.processed_data)*100:.1f}%)")
        report.append(f"Average rating of high-rated: {high_rated['rating'].mean():.2f}")
        report.append(f"Average members for high-rated: {high_rated['members'].mean():,.0f}")
        report.append("")
        
        # Top anime
        if len(high_rated) > 0:
            report.append("🏆 TOP 10 HIGHEST RATED ANIME")
            report.append("-" * 30)
            top_anime = high_rated.nlargest(10, 'rating')
            for idx, row in top_anime.iterrows():
                report.append(f"{row['rating']:4.2f} - {row['name'][:40]:40} ({row['type']})")
            report.append("")
        
        # System info
        if SYSTEM_AVAILABLE and self.recommendation_system:
            report.append("🔧 SYSTEM INFORMATION")
            report.append("-" * 30)
            report.append("Recommendation system: Available")
            report.append(f"Content recommender: {'Built' if self.recommendation_system.tfidf_matrix is not None else 'Not built'}")
            report.append(f"Clustering model: {'Built' if 'cluster' in self.processed_data.columns else 'Not built'}")
        else:
            report.append("🔧 SYSTEM INFORMATION")
            report.append("-" * 30)
            report.append("Recommendation system: Not available")
            report.append("Running in basic visualization mode")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        with open("anime_simple_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print("✅ Summary report saved as 'anime_simple_report.txt'")
        print("\n" + report_text)
    
    def run_visualization_suite(self):
        """Run all available visualizations."""
        print("\n" + "🎨" * 20)
        print("   SIMPLE ANIME VISUALIZATION SUITE")
        print("🎨" * 20)
        
        try:
            self.create_basic_overview()
            print("\n" + "="*50)
            
            self.create_genre_analysis()
            print("\n" + "="*50)
            
            self.create_rating_analysis()
            print("\n" + "="*50)
            
            if SYSTEM_AVAILABLE:
                self.create_recommendation_test()
                print("\n" + "="*50)
                
                self.create_clustering_analysis()
                print("\n" + "="*50)
            
            self.generate_summary_report()
            
            # Summary
            print("\n🎉 VISUALIZATION SUITE COMPLETED!")
            print("="*50)
            print("📁 Generated Files:")
            
            files = [
                "anime_overview.png",
                "genre_analysis.png",
                "rating_analysis.png",
                "anime_simple_report.txt"
            ]
            
            if SYSTEM_AVAILABLE:
                files.extend([
                    "recommendation_test.png",
                    "clustering_analysis.png"
                ])
            
            for i, file in enumerate(files, 1):
                print(f"   {i:2d}. {file}")
            
            print(f"\n✨ Successfully generated {len(files)} files!")
            
        except Exception as e:
            print(f"❌ Error in visualization suite: {str(e)}")
            raise

def main():
    """Main function to run the visualization system."""
    print("🎨 SIMPLE ANIME RECOMMENDATION VISUALIZER")
    print("🤖 Basic Analytics & Visualization Engine")
    print("="*50)
    
    try:
        # Initialize visualization engine
        viz_engine = SimpleAnimeVisualizer("anime.csv")
        
        # Interactive menu
        while True:
            print("\n" + "="*40)
            print("🎯 VISUALIZATION OPTIONS")
            print("="*40)
            print("1. 📊 Basic Dataset Overview")
            print("2. 🎭 Genre Analysis")
            print("3. 📈 Rating Analysis")
            print("4. 🎯 Test Recommendations")
            print("5. 🔬 Clustering Analysis")
            print("6. 📋 Generate Summary Report")
            print("7. 🎨 RUN ALL VISUALIZATIONS")
            print("8. 🚪 Exit")
            
            choice = input("\n🎮 Enter your choice (1-8): ").strip()
            
            if choice == '1':
                viz_engine.create_basic_overview()
            elif choice == '2':
                viz_engine.create_genre_analysis()
            elif choice == '3':
                viz_engine.create_rating_analysis()
            elif choice == '4':
                viz_engine.create_recommendation_test()
            elif choice == '5':
                viz_engine.create_clustering_analysis()
            elif choice == '6':
                viz_engine.generate_summary_report()
            elif choice == '7':
                viz_engine.run_visualization_suite()
            elif choice == '8':
                print("\n🎌 Thank you for using the Simple Anime Visualizer!")
                print("✨ Happy analyzing! 👋")
                break
            else:
                print("❌ Invalid choice. Please enter 1-8.")
                
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        print("Please check your anime.csv file and paths.")

if __name__ == "__main__":
    main()