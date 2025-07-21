# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# def clean_and_reduce_dataset(input_file, output_file, target_size=3000):
#     """
#     Clean the anime dataset and reduce it to the target size while maintaining quality and diversity
#     """
#     print(f"Loading dataset from {input_file}...")
    
#     # Load the dataset
#     try:
#         df = pd.read_csv(input_file)
#         print(f"Original dataset shape: {df.shape}")
#         print(f"Original dataset info:")
#         print(df.info())
#         print(f"\nMissing values:\n{df.isnull().sum()}")
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         return
    
#     print("\n" + "="*60)
#     print("STEP 1: DATA CLEANING")
#     print("="*60)
    
#     # 1. Remove rows with missing critical information
#     original_size = len(df)
    
#     # Remove rows with missing name (critical)
#     df = df.dropna(subset=['name'])
#     print(f"After removing rows with missing names: {len(df)} rows")
    
#     # Remove rows with missing rating (important for recommendations)
#     df = df.dropna(subset=['rating'])
#     print(f"After removing rows with missing ratings: {len(df)} rows")
    
#     # 2. Handle missing values in other columns
#     df['genre'] = df['genre'].fillna('Unknown')
#     df['type'] = df['type'].fillna('Unknown')
#     df['episodes'] = df['episodes'].fillna(1)  # Default to 1 episode
#     df['members'] = df['members'].fillna(0)
    
#     print(f"Missing values after cleaning:\n{df.isnull().sum()}")
    
#     # 3. Remove duplicates based on name (case-insensitive)
#     before_dup = len(df)
#     df['name_lower'] = df['name'].str.lower().str.strip()
#     df = df.drop_duplicates(subset=['name_lower'])
#     df = df.drop('name_lower', axis=1)
#     print(f"After removing duplicates: {len(df)} rows (removed {before_dup - len(df)} duplicates)")
    
#     # 4. Data validation and cleaning
#     # Remove invalid ratings (should be between 0-10)
#     df = df[(df['rating'] >= 0) & (df['rating'] <= 10)]
#     print(f"After removing invalid ratings: {len(df)} rows")
    
#     # Convert episodes to numeric
#     df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

#     # (Optional) Remove rows with missing episodes
#     df = df[df['episodes'].notnull()]

#     # Now you can safely compare
#     df = df[df['episodes'] >= 0]
#     print(f"After removing negative episodes/members: {len(df)} rows")
    
#     # 5. Clean genre column - fix HTML entities
#     df['genre'] = df['genre'].str.replace('&#039;', "'", regex=False)
#     df['genre'] = df['genre'].str.replace('&amp;', '&', regex=False)
    
#     # 6. Clean name column - fix HTML entities
#     df['name'] = df['name'].str.replace('&#039;', "'", regex=False)
#     df['name'] = df['name'].str.replace('&amp;', '&', regex=False)
    
#     print("\n" + "="*60)
#     print("STEP 2: FEATURE ENGINEERING FOR SELECTION")
#     print("="*60)
    
#     # Create features for intelligent selection
#     df['genre_count'] = df['genre'].apply(lambda x: len(str(x).split(', ')) if pd.notna(x) and str(x) != 'Unknown' else 0)
#     df['popularity_score'] = df['members'] * df['rating']
#     df['episode_category'] = pd.cut(df['episodes'], 
#                                    bins=[0, 1, 12, 26, 50, float('inf')], 
#                                    labels=['Movie/Special', 'Short', 'Standard', 'Long', 'Very Long'])
    
#     print(f"Dataset after feature engineering: {df.shape}")
    
#     print("\n" + "="*60)
#     print("STEP 3: INTELLIGENT DATA REDUCTION")
#     print("="*60)
    
#     if len(df) <= target_size:
#         print(f"Dataset already has {len(df)} rows, which is <= target size of {target_size}")
#         final_df = df.copy()
#     else:
#         # Strategy: Maintain diversity while prioritizing quality
        
#         # 1. Always keep the top-rated animes (top 10%)
#         top_rated_count = int(target_size * 0.1)  # 10% for top-rated
#         top_rated = df.nlargest(top_rated_count, 'rating')
#         print(f"Selected {len(top_rated)} top-rated animes")
        
#         # 2. Select popular animes (high members count) - 20%
#         remaining_df = df[~df.index.isin(top_rated.index)]
#         popular_count = int(target_size * 0.2)  # 20% for popular
#         popular = remaining_df.nlargest(popular_count, 'members')
#         print(f"Selected {len(popular)} popular animes")
        
#         # 3. Ensure type diversity - 30%
#         remaining_df = remaining_df[~remaining_df.index.isin(popular.index)]
#         type_diversity_count = int(target_size * 0.3)  # 30% for type diversity
        
#         # Sample proportionally from each type
#         type_counts = remaining_df['type'].value_counts()
#         type_samples = []
        
#         for anime_type in type_counts.index:
#             type_data = remaining_df[remaining_df['type'] == anime_type]
#             # Calculate sample size proportional to type frequency
#             sample_size = min(
#                 len(type_data),
#                 max(1, int(type_diversity_count * (type_counts[anime_type] / type_counts.sum())))
#             )
#             if sample_size > 0:
#                 # Sample the best rated from this type
#                 type_sample = type_data.nlargest(sample_size, 'rating')
#                 type_samples.append(type_sample)
        
#         type_diversity = pd.concat(type_samples) if type_samples else pd.DataFrame()
#         print(f"Selected {len(type_diversity)} animes for type diversity")
        
#         # 4. Genre diversity - 25%
#         remaining_df = remaining_df[~remaining_df.index.isin(type_diversity.index)]
#         genre_diversity_count = int(target_size * 0.25)  # 25% for genre diversity
        
#         # Get all unique genres
#         all_genres = set()
#         for genres in remaining_df['genre'].dropna():
#             if str(genres) != 'Unknown':
#                 all_genres.update([g.strip() for g in str(genres).split(',')])
        
#         # Sample animes to cover different genres
#         genre_samples = []
#         for genre in list(all_genres)[:20]:  # Top 20 genres
#             genre_data = remaining_df[remaining_df['genre'].str.contains(genre, na=False)]
#             if len(genre_data) > 0:
#                 # Take top 2 from each genre
#                 genre_sample = genre_data.nlargest(min(2, len(genre_data)), 'rating')
#                 genre_samples.append(genre_sample)
        
#         genre_diversity = pd.concat(genre_samples) if genre_samples else pd.DataFrame()
#         genre_diversity = genre_diversity.drop_duplicates().head(genre_diversity_count)
#         print(f"Selected {len(genre_diversity)} animes for genre diversity")
        
#         # 5. Random sampling for the remaining slots - 15%
#         remaining_df = remaining_df[~remaining_df.index.isin(genre_diversity.index)]
#         remaining_count = target_size - len(top_rated) - len(popular) - len(type_diversity) - len(genre_diversity)
        
#         if remaining_count > 0 and len(remaining_df) > 0:
#             # Weighted random sampling (higher weight for better ratings)
#             weights = remaining_df['rating'] / remaining_df['rating'].sum()
#             random_sample = remaining_df.sample(
#                 n=min(remaining_count, len(remaining_df)), 
#                 weights=weights,
#                 random_state=42
#             )
#             print(f"Selected {len(random_sample)} animes through weighted random sampling")
#         else:
#             random_sample = pd.DataFrame()
        
#         # Combine all selections
#         final_df = pd.concat([top_rated, popular, type_diversity, genre_diversity, random_sample])
#         final_df = final_df.drop_duplicates()  # Remove any duplicates
        
#         # If still over target, randomly remove excess
#         if len(final_df) > target_size:
#             final_df = final_df.sample(n=target_size, random_state=42)
    
#     # Clean up temporary columns
#     columns_to_remove = ['genre_count', 'popularity_score', 'episode_category']
#     final_df = final_df.drop(columns=[col for col in columns_to_remove if col in final_df.columns])
    
#     # Reset index
#     final_df = final_df.reset_index(drop=True)
    
#     print("\n" + "="*60)
#     print("STEP 4: FINAL DATASET STATISTICS")
#     print("="*60)
    
#     print(f"Final dataset shape: {final_df.shape}")
#     print(f"Reduction: {original_size} → {len(final_df)} rows ({(original_size-len(final_df))/original_size*100:.1f}% reduction)")
    
#     print(f"\nFinal dataset statistics:")
#     print(f"Rating range: {final_df['rating'].min():.2f} - {final_df['rating'].max():.2f}")
#     print(f"Average rating: {final_df['rating'].mean():.2f}")
#     print(f"Average episodes: {final_df['episodes'].mean():.1f}")
#     print(f"Average members: {final_df['members'].mean():.0f}")
    
#     print(f"\nType distribution:")
#     print(final_df['type'].value_counts())
    
#     print(f"\nRating distribution:")
#     print(pd.cut(final_df['rating'], bins=5).value_counts().sort_index())
    
#     # Save the cleaned dataset
#     print(f"\nSaving cleaned dataset to {output_file}...")
#     final_df.to_csv(output_file, index=False)
#     print(f"Dataset successfully saved!")
    
#     return final_df

# def main():
#     """Main function to run the data cleaning process"""
#     input_file = r"D:\AI assignment\anime.csv"
#     output_file = r"D:\AI assignment\anime_cleaned.csv"
#     target_size = 3000
    
#     print("Anime Dataset Cleaning and Reduction Tool")
#     print("=" * 60)
#     print(f"Input file: {input_file}")
#     print(f"Output file: {output_file}")
#     print(f"Target size: {target_size} rows")
#     print("=" * 60)
    
#     # Run the cleaning process
#     cleaned_df = clean_and_reduce_dataset(input_file, output_file, target_size)
    
#     if cleaned_df is not None:
#         print(f"\n✅ Process completed successfully!")
#         print(f"Cleaned dataset saved as: {output_file}")
#         print(f"Final size: {len(cleaned_df)} rows")
        
#         # Show a sample of the cleaned data
#         print(f"\nSample of cleaned data:")
#         print(cleaned_df.head(10).to_string(index=False))
#     else:
#         print("❌ Process failed!")

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

class AnimeRecommendationSystem:
    """
    A hybrid anime recommendation system combining collaborative filtering 
    and content-based filtering approaches.
    """
    
    def __init__(self):
        self.anime_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.content_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.trained = False
        
    def load_and_preprocess_data(self, anime_path, ratings_path):
        """
        Load and preprocess the anime and ratings datasets
        """
        print("Loading datasets...")
        
        # Load datasets
        self.anime_df = pd.read_csv(anime_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        print(f"Anime dataset shape: {self.anime_df.shape}")
        print(f"Ratings dataset shape: {self.ratings_df.shape}")
        
        # Data preprocessing for anime dataset
        print("\nPreprocessing anime dataset...")
        
        # Handle missing values
        self.anime_df['genre'] = self.anime_df['genre'].fillna('Unknown')
        self.anime_df['type'] = self.anime_df['type'].fillna('Unknown')
        self.anime_df['episodes'] = self.anime_df['episodes'].fillna(0)
        self.anime_df['rating'] = self.anime_df['rating'].fillna(self.anime_df['rating'].mean())
        self.anime_df['members'] = self.anime_df['members'].fillna(0)
        
        # Clean episode data (convert 'Unknown' to 0)
        self.anime_df['episodes'] = pd.to_numeric(self.anime_df['episodes'], errors='coerce').fillna(0)
        
        # Data preprocessing for ratings dataset
        print("Preprocessing ratings dataset...")
        
        # Remove ratings of -1 (which typically means the user watched but didn't rate)
        self.ratings_df = self.ratings_df[self.ratings_df['rating'] != -1]
        
        # Filter out users with very few ratings (less than 5)
        user_counts = self.ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        self.ratings_df = self.ratings_df[self.ratings_df['user_id'].isin(valid_users)]
        
        # Filter out anime with very few ratings (less than 10)
        anime_counts = self.ratings_df['anime_id'].value_counts()
        valid_anime = anime_counts[anime_counts >= 10].index
        self.ratings_df = self.ratings_df[self.ratings_df['anime_id'].isin(valid_anime)]
        
        # Ensure we only have anime in ratings that exist in anime dataset
        self.ratings_df = self.ratings_df[self.ratings_df['anime_id'].isin(self.anime_df['anime_id'])]
        
        print(f"After preprocessing - Ratings shape: {self.ratings_df.shape}")
        print(f"Unique users: {self.ratings_df['user_id'].nunique()}")
        print(f"Unique anime: {self.ratings_df['anime_id'].nunique()}")
        
        # Create user-item matrix
        self.create_user_item_matrix()
        
    def create_user_item_matrix(self):
        """Create user-item matrix for collaborative filtering"""
        print("Creating user-item matrix...")
        
        self.user_item_matrix = self.ratings_df.pivot(
            index='user_id', 
            columns='anime_id', 
            values='rating'
        ).fillna(0)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        
    def create_content_features(self):
        """Create content-based features using genres and other metadata"""
        print("Creating content-based features...")
        
        # Combine text features
        self.anime_df['content_features'] = (
            self.anime_df['genre'].astype(str) + ' ' +
            self.anime_df['type'].astype(str)
        )
        
        # Create TF-IDF vectors for genres and type
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.anime_df['content_features'])
        
        # Add numerical features
        numerical_features = self.anime_df[['rating', 'episodes', 'members']].copy()
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine TF-IDF and numerical features
        content_features = np.hstack([tfidf_matrix.toarray(), numerical_features])
        
        # Calculate content similarity matrix
        self.content_similarity_matrix = cosine_similarity(content_features)
        
        print(f"Content similarity matrix shape: {self.content_similarity_matrix.shape}")
        
    def collaborative_filtering_recommendations(self, user_id, n_recommendations=10):
        """
        Generate recommendations using collaborative filtering (user-based)
        """
        if user_id not in self.user_item_matrix.index:
            return []
            
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Find similar users
        user_similarities = {}
        for other_user in self.user_item_matrix.index:
            if other_user != user_id:
                # Calculate cosine similarity
                user_vector = self.user_item_matrix.loc[user_id].values
                other_user_vector = self.user_item_matrix.loc[other_user].values
                
                # Only consider users who have rated common items
                common_items = (user_vector != 0) & (other_user_vector != 0)
                
                if np.sum(common_items) > 0:
                    similarity = 1 - cosine(user_vector[common_items], 
                                          other_user_vector[common_items])
                    if not np.isnan(similarity):
                        user_similarities[other_user] = similarity
        
        # Get top similar users
        similar_users = sorted(user_similarities.items(), 
                             key=lambda x: x[1], reverse=True)[:50]
        
        # Generate recommendations
        recommendations = {}
        user_rated_anime = set(user_ratings[user_ratings > 0].index)
        
        for similar_user, similarity in similar_users:
            similar_user_ratings = self.user_item_matrix.loc[similar_user]
            
            for anime_id in similar_user_ratings.index:
                if (anime_id not in user_rated_anime and 
                    similar_user_ratings[anime_id] > 0):
                    
                    if anime_id not in recommendations:
                        recommendations[anime_id] = 0
                    
                    recommendations[anime_id] += similarity * similar_user_ratings[anime_id]
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(recommendations.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return [anime_id for anime_id, _ in sorted_recommendations[:n_recommendations]]
    
    def content_based_recommendations(self, anime_id, n_recommendations=10):
        """
        Generate recommendations using content-based filtering
        """
        if anime_id not in self.anime_df['anime_id'].values:
            return []
            
        # Find the index of the anime
        anime_idx = self.anime_df[self.anime_df['anime_id'] == anime_id].index[0]
        
        # Get similarity scores for this anime
        sim_scores = list(enumerate(self.content_similarity_matrix[anime_idx]))
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the indices of the most similar anime (excluding the anime itself)
        similar_anime_indices = [i for i, _ in sim_scores[1:n_recommendations+1]]
        
        # Return the anime_ids
        return self.anime_df.iloc[similar_anime_indices]['anime_id'].tolist()
    
    def hybrid_recommendations(self, user_id, n_recommendations=10, 
                             cf_weight=0.6, cb_weight=0.4):
        """
        Generate hybrid recommendations combining collaborative and content-based filtering
        """
        # Get collaborative filtering recommendations
        cf_recommendations = self.collaborative_filtering_recommendations(
            user_id, n_recommendations * 2
        )
        
        # Get content-based recommendations based on user's highly rated anime
        cb_recommendations = []
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            highly_rated_anime = user_ratings[user_ratings >= 8].index.tolist()
            
            for anime_id in highly_rated_anime[:3]:  # Take top 3 highly rated
                cb_recs = self.content_based_recommendations(anime_id, n_recommendations)
                cb_recommendations.extend(cb_recs)
        
        # Combine recommendations with weights
        recommendation_scores = {}
        
        # Add collaborative filtering scores
        for i, anime_id in enumerate(cf_recommendations):
            score = cf_weight * (len(cf_recommendations) - i) / len(cf_recommendations)
            recommendation_scores[anime_id] = score
        
        # Add content-based scores
        for i, anime_id in enumerate(cb_recommendations):
            cb_score = cb_weight * (len(cb_recommendations) - i) / len(cb_recommendations)
            if anime_id in recommendation_scores:
                recommendation_scores[anime_id] += cb_score
            else:
                recommendation_scores[anime_id] = cb_score
        
        # Sort by combined scores
        sorted_recommendations = sorted(recommendation_scores.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return [anime_id for anime_id, _ in sorted_recommendations[:n_recommendations]]
    
    def train_model(self):
        """Train the recommendation model"""
        print("Training recommendation model...")
        
        # Create content-based features
        self.create_content_features()
        
        self.trained = True
        print("Model training completed!")
    
    def evaluate_model(self, test_size=0.2):
        """
        Evaluate the model using RMSE and MAE on a test set
        """
        print("Evaluating model...")
        
        # Create train-test split
        train_ratings, test_ratings = train_test_split(
            self.ratings_df, test_size=test_size, random_state=42
        )
        
        # Simple baseline: predict average rating for each anime
        anime_avg_ratings = train_ratings.groupby('anime_id')['rating'].mean()
        
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            anime_id = row['anime_id']
            actual_rating = row['rating']
            
            # Predict using anime average rating
            if anime_id in anime_avg_ratings:
                predicted_rating = anime_avg_ratings[anime_id]
            else:
                predicted_rating = train_ratings['rating'].mean()
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        
        return {'RMSE': rmse, 'MAE': mae}
    
    def get_anime_info(self, anime_id):
        """Get anime information by ID"""
        anime_info = self.anime_df[self.anime_df['anime_id'] == anime_id]
        if not anime_info.empty:
            return anime_info.iloc[0].to_dict()
        return None
    
    def display_recommendations(self, recommendations, title="Recommendations"):
        """Display recommendations with anime information"""
        print(f"\n{title}:")
        print("-" * 80)
        
        for i, anime_id in enumerate(recommendations, 1):
            anime_info = self.get_anime_info(anime_id)
            if anime_info:
                print(f"{i}. {anime_info['name']}")
                print(f"   Genre: {anime_info['genre']}")
                print(f"   Type: {anime_info['type']} | Episodes: {anime_info['episodes']}")
                print(f"   Rating: {anime_info['rating']:.2f} | Members: {anime_info['members']:,}")
                print()

def main():
    """Main function to demonstrate the recommendation system"""
    
    # Initialize the recommendation system
    rec_system = AnimeRecommendationSystem()
    
    # Note: In a real implementation, you would use actual file paths
    print("=== ANIME RECOMMENDATION SYSTEM ===")
    print("This is a demonstration of the system architecture.")
    print("In actual implementation, replace with your CSV file paths:\n")
    
    print("# Load and preprocess data")
    print("rec_system.load_and_preprocess_data('anime.csv', 'rating.csv')")
    
    print("\n# Train the model")
    print("rec_system.train_model()")
    
    print("\n# Evaluate the model")
    print("evaluation_results = rec_system.evaluate_model()")
    
    print("\n# Get recommendations for a user")
    print("user_id = 1")
    print("recommendations = rec_system.hybrid_recommendations(user_id, n_recommendations=10)")
    print("rec_system.display_recommendations(recommendations, 'Hybrid Recommendations for User 1')")
    
    print("\n# Get content-based recommendations for an anime")
    print("anime_id = 32281  # Kimi no Na wa")
    print("content_recs = rec_system.content_based_recommendations(anime_id, n_recommendations=5)")
    print("rec_system.display_recommendations(content_recs, 'Similar to Kimi no Na wa')")
    
    # Create sample data visualization code
    print("\n# Visualization functions")
    print("""
def create_visualizations(rec_system):
    plt.figure(figsize=(15, 10))
    
    # 1. Rating distribution
    plt.subplot(2, 3, 1)
    plt.hist(rec_system.anime_df['rating'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Anime Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    
    # 2. Top genres
    plt.subplot(2, 3, 2)
    genres = []
    for genre_str in rec_system.anime_df['genre']:
        if pd.notna(genre_str):
            genres.extend([g.strip() for g in genre_str.split(',')])
    
    genre_counts = pd.Series(genres).value_counts().head(10)
    genre_counts.plot(kind='bar')
    plt.title('Top 10 Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 3. Type distribution
    plt.subplot(2, 3, 3)
    rec_system.anime_df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Anime Type Distribution')
    
    # 4. Episodes vs Rating scatter plot
    plt.subplot(2, 3, 4)
    plt.scatter(rec_system.anime_df['episodes'], rec_system.anime_df['rating'], alpha=0.5)
    plt.title('Episodes vs Rating')
    plt.xlabel('Episodes')
    plt.ylabel('Rating')
    
    # 5. Members vs Rating
    plt.subplot(2, 3, 5)
    plt.scatter(rec_system.anime_df['members'], rec_system.anime_df['rating'], alpha=0.5)
    plt.title('Members vs Rating')
    plt.xlabel('Members')
    plt.ylabel('Rating')
    plt.xscale('log')
    
    # 6. Rating distribution by type
    plt.subplot(2, 3, 6)
    sns.boxplot(data=rec_system.anime_df, x='type', y='rating')
    plt.title('Rating Distribution by Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Usage:
# create_visualizations(rec_system)
    """)

if __name__ == "__main__":
    main()