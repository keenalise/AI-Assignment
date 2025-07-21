import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class AnimeRecommender:
    def __init__(self, anime_file='anime.csv', ratings_file='rating.csv'):
        # Load and preprocess data
        self.anime_df = pd.read_csv(anime_file)
        self.ratings_df = pd.read_csv(ratings_file)
        self._preprocess_data()
        self._build_matrices()
        
    def _preprocess_data(self):
        # Clean anime data
        self.anime_df['genre'] = self.anime_df['genre'].fillna('').str.replace(', ', ',')
        self.anime_df['rating'] = self.anime_df['rating'].fillna(self.anime_df['rating'].mean())
        self.anime_df['episodes'] = pd.to_numeric(self.anime_df['episodes'], errors='coerce').fillna(1)
        
        # Clean ratings data
        self.ratings_df = self.ratings_df[self.ratings_df['rating'] != -1]  # Remove unrated entries
        self.ratings_df['rating'] = self.ratings_df['rating'].clip(0, 10)
        
        # Normalize ratings
        self.scaler = MinMaxScaler()
        self.anime_df['normalized_rating'] = self.scaler.fit_transform(
            self.anime_df[['rating']].values
        )
        
    def _build_matrices(self):
        # Remove duplicate (user_id, anime_id) pairs by taking the mean rating
        self.ratings_df = self.ratings_df.groupby(['user_id', 'anime_id'], as_index=False)['rating'].mean()

        # Now pivot
        self.user_item_matrix = self.ratings_df.pivot(
            index='user_id', 
            columns='anime_id', 
            values='rating'
        ).fillna(0)
        
        # Create genre-based feature matrix for content-based filtering
        self.tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(','))
        self.genre_matrix = self.tfidf.fit_transform(self.anime_df['genre'])
        
    def collaborative_filtering(self, user_id, n_recommendations=5):
        # Apply SVD for matrix factorization
        svd = TruncatedSVD(n_components=50, random_state=42)
        matrix = svd.fit_transform(self.user_item_matrix)
        user_ratings = svd.transform(self.user_item_matrix.loc[user_id].values.reshape(1, -1))
        
        # Calculate correlation
        corr_matrix = np.corrcoef(matrix)
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similar_users = corr_matrix[user_idx]
        
        # Get top similar users
        similar_users_idx = np.argsort(similar_users)[-n_recommendations-1:-1]
        similar_users_scores = similar_users[similar_users_idx]
        
        # Generate recommendations
        recommendations = []
        for idx in similar_users_idx:
            user_ratings = self.user_item_matrix.iloc[idx]
            top_anime = user_ratings[user_ratings > 0].sort_values(ascending=False).index[:n_recommendations]
            recommendations.extend(top_anime)
        
        return recommendations[:n_recommendations]
    
    def content_based_filtering(self, anime_id, n_recommendations=5):
        # Calculate cosine similarity between anime
        cosine_sim = cosine_similarity(self.genre_matrix)
        anime_idx = self.anime_df.index[self.anime_df['anime_id'] == anime_id].tolist()[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[anime_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        anime_indices = [i[0] for i in sim_scores]
        return self.anime_df['anime_id'].iloc[anime_indices].tolist()
    
    def hybrid_recommendation(self, user_id, n_recommendations=5):
        # Get collaborative filtering recommendations
        collab_recs = self.collaborative_filtering(user_id, n_recommendations)
        
        # Get content-based recommendations for each collaborative recommendation
        final_recs = set()
        for anime_id in collab_recs:
            content_recs = self.content_based_filtering(anime_id, n_recommendations//2)
            final_recs.update(content_recs)
        
        # Combine and rank recommendations
        rec_df = self.anime_df[self.anime_df['anime_id'].isin(final_recs)]
        rec_df = rec_df.sort_values('normalized_rating', ascending=False)
        
        # Return top recommendations with details
        return rec_df[['anime_id', 'name', 'genre', 'rating']].head(n_recommendations).to_dict('records')

    def evaluate(self):
        # Calculate RMSE for collaborative filtering
        predictions = []
        actual = []
        
        for _, row in self.ratings_df.iterrows():
            user_id, anime_id, rating = row['user_id'], row['anime_id'], row['rating']
            try:
                user_idx = self.user_item_matrix.index.get_loc(user_id)
                anime_idx = self.user_item_matrix.columns.get_loc(anime_id)
                pred = np.dot(
                    self.user_item_matrix.iloc[user_idx],
                    self.user_item_matrix.columns == anime_id
                )
                predictions.append(pred)
                actual.append(rating)
            except:
                continue
                
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actual))**2))
        return {'RMSE': rmse}

# Example usage
if __name__ == "__main__":
    recommender = AnimeRecommender()
    user_id = 1
    recommendations = recommender.hybrid_recommendation(user_id, 5)
    print("Recommendations for user", user_id)
    for rec in recommendations:
        print(f"{rec['name']} (Rating: {rec['rating']}) - Genres: {rec['genre']}")
    print("\nModel Evaluation:", recommender.evaluate())