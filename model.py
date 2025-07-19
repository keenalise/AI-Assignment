import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

class AnimeRecommendationSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.genre_encoder = MultiLabelBinarizer()
        self.model = None
        self.feature_vectors = None
        self.df = None
        
    def load_data(self, csv_path):
        """Load anime dataset from CSV file"""
        self.df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {len(self.df)} anime entries")
        return self.df
    
    def preprocess_features(self, df):
        """Extract and preprocess features for the model"""
        # Handle missing values
        df = df.dropna(subset=['genre', 'rating', 'episodes'])
        
        # Process genres (split by comma and clean)
        genre_lists = []
        for genre_str in df['genre']:
            if pd.isna(genre_str):
                genre_lists.append([])
            else:
                genres = [g.strip() for g in str(genre_str).split(',')]
                genre_lists.append(genres)
        
        # Encode genres using MultiLabelBinarizer
        genre_encoded = self.genre_encoder.fit_transform(genre_lists)
        
        # Prepare numerical features
        numerical_features = df[['rating', 'episodes']].values
        numerical_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine all features
        feature_matrix = np.hstack([genre_encoded, numerical_scaled])
        
        return feature_matrix, df
    
    def build_autoencoder(self, input_dim, encoding_dim=32):
        """Build an autoencoder for learning anime representations"""
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Full autoencoder
        autoencoder = keras.Model(input_layer, decoded)
        
        # Encoder model for getting embeddings
        encoder = keras.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder, encoder
    
    def train_model(self, df=None, encoding_dim=32, epochs=100, validation_split=0.2):
        """Train the recommendation model"""
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("No dataset provided. Use load_data() first or pass df parameter.")
        
        # Preprocess features
        feature_matrix, processed_df = self.preprocess_features(df)
        self.df = processed_df
        self.feature_vectors = feature_matrix
        
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Number of unique genres: {len(self.genre_encoder.classes_)}")
        
        # Build model
        autoencoder, encoder = self.build_autoencoder(feature_matrix.shape[1], encoding_dim)
        self.model = encoder
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        # Train autoencoder
        history = autoencoder.fit(
            feature_matrix, feature_matrix,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Generate embeddings for all anime
        self.anime_embeddings = self.model.predict(feature_matrix)
        
        print("Training completed successfully!")
        return history
    
    def encode_user_input(self, genre, rating, episodes):
        """Convert user input to feature vector"""
        # Handle genre input
        if isinstance(genre, str):
            genre_list = [g.strip() for g in genre.split(',')]
        else:
            genre_list = genre
        
        # Encode genre
        genre_encoded = self.genre_encoder.transform([genre_list])
        
        # Scale numerical features
        numerical_features = np.array([[float(rating), float(episodes)]])
        numerical_scaled = self.scaler.transform(numerical_features)
        
        # Combine features
        feature_vector = np.hstack([genre_encoded, numerical_scaled])
        
        return feature_vector
    
    def get_recommendations(self, genre, rating, episodes, n_recommendations=5, method='cosine'):
        """Get anime recommendations based on user preferences"""
        if self.model is None or self.feature_vectors is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Encode user input
        user_vector = self.encode_user_input(genre, rating, episodes)
        
        if method == 'embedding':
            # Use neural network embeddings
            user_embedding = self.model.predict(user_vector)
            similarities = cosine_similarity(user_embedding, self.anime_embeddings)[0]
        else:
            # Use direct feature similarity
            similarities = cosine_similarity(user_vector, self.feature_vectors)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            anime_info = self.df.iloc[idx]
            recommendations.append({
                'name': anime_info['name'],
                'genre': anime_info['genre'],
                'rating': anime_info['rating'],
                'episodes': anime_info['episodes'],
                'type': anime_info.get('type', 'Unknown'),
                'similarity_score': similarities[idx]
            })
        
        return recommendations
    
    def find_similar_anime(self, anime_name, n_recommendations=5):
        """Find anime similar to a given anime"""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        # Find the anime in dataset
        anime_match = self.df[self.df['name'].str.contains(anime_name, case=False, na=False)]
        
        if anime_match.empty:
            print(f"Anime '{anime_name}' not found in dataset.")
            return []
        
        # Get the first match
        target_anime = anime_match.iloc[0]
        
        # Get recommendations based on this anime's features
        return self.get_recommendations(
            target_anime['genre'], 
            target_anime['rating'], 
            target_anime['episodes'],
            n_recommendations + 1  # +1 to exclude the anime itself
        )[1:]  # Exclude the first result (the anime itself)
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'genre_encoder': self.genre_encoder,
            'feature_vectors': self.feature_vectors,
            'anime_embeddings': getattr(self, 'anime_embeddings', None),
            'df': self.df
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.genre_encoder = model_data['genre_encoder']
        self.feature_vectors = model_data['feature_vectors']
        self.anime_embeddings = model_data.get('anime_embeddings')
        self.df = model_data['df']
        print("Model loaded successfully!")

# Example usage with your dataset
def main():
    """Example usage of the recommendation system with your dataset"""
    
    # Initialize the system
    recommender = AnimeRecommendationSystem()
    
    # Load your dataset
    dataset_path = r"D:\AI assignment\anime_cleaned.csv"
    
    try:
        df = recommender.load_data(dataset_path)
        print(f"Successfully loaded dataset with {len(df)} anime entries")
        print(f"Dataset columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        print("Creating sample dataset for demonstration...")
        
        # Fallback to sample data if file not found
        sample_data = {
            'anime_id': [33662, 23005, 32281, 33607, 5114, 28977, 26313, 9253, 9969, 32935],
            'name': [
                'Taka no Tsume 8: Yoshida-kun no X-Files',
                'Mogura no Motoro',
                'Kimi no Na wa.',
                'Kahei no Umi',
                'Fullmetal Alchemist: Brotherhood',
                'Gintama°',
                'Yakusoku: Africa Mizu to Midori',
                'Steins;Gate',
                "Gintama'",
                'Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou'
            ],
            'genre': [
                'Comedy, Parody',
                'Slice of Life',
                'Drama, Romance, School, Supernatural',
                'Historical',
                'Action, Adventure, Drama, Fantasy, Magic, Military, Shounen',
                'Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen',
                'Drama, Kids',
                'Sci-Fi, Thriller',
                'Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen',
                'Comedy, Drama, School, Shounen, Sports'
            ],
            'type': ['Movie', 'Movie', 'Movie', 'Movie', 'TV', 'TV', 'OVA', 'TV', 'TV', 'TV'],
            'episodes': [1.0, 1.0, 1.0, 1.0, 64.0, 51.0, 1.0, 24.0, 51.0, 10.0],
            'rating': [10.0, 9.5, 9.37, 9.33, 9.26, 9.25, 9.25, 9.17, 9.16, 9.15],
            'members': [13, 62, 200630, 44, 793665, 114262, 53, 673572, 151266, 93351]
        }
        df = pd.DataFrame(sample_data)
    
    # Train the model
    print("\n=== Training the recommendation model ===")
    recommender.train_model(df, epochs=100)
    
    # Save the trained model for future use
    recommender.save_model("anime_recommender_model.pkl")
    
    # Get recommendations based on user preferences
    print("\n=== Recommendations based on preferences ===")
    recommendations = recommender.get_recommendations(
        genre="Action, Comedy, Shounen",
        rating=9.0,
        episodes=25,
        n_recommendations=3
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Genre: {rec['genre']}")
        print(f"   Rating: {rec['rating']}, Episodes: {rec['episodes']}")
        print(f"   Similarity: {rec['similarity_score']:.4f}\n")
    
    # Find similar anime to a specific one
    print("=== Similar to Steins;Gate ===")
    similar = recommender.find_similar_anime("Steins", n_recommendations=2)
    
    for i, rec in enumerate(similar, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Genre: {rec['genre']}")
        print(f"   Similarity: {rec['similarity_score']:.4f}\n")

if __name__ == "__main__":
    main()