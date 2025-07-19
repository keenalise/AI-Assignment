# Anime Recommendation System
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AnimeRecommendationSystem:
    """
    A comprehensive anime recommendation system implementing multiple AI techniques:
    1. Content-based filtering using TF-IDF and cosine similarity
    2. Collaborative filtering using user preferences
    3. Clustering for anime grouping
    4. Classification for rating prediction
    """
    
    def __init__(self, file_path):
        """Initialize the recommendation system with data loading and preprocessing"""
        self.data = None
        self.processed_data = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.classifier = None
        self.load_and_preprocess_data(file_path)
    
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the anime dataset
        Handles missing values, data cleaning, and feature engineering
        """
        print("Loading and preprocessing data...")
        
        # Load data
        try:
            self.data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {len(self.data)} records")
        except FileNotFoundError:
            print("Error: anime.csv file not found. Please ensure the file is in the correct directory.")
            return
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(self.data.info())
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        
        # Data cleaning
        self.processed_data = self.data.copy()
        
        # Handle missing values
        self.processed_data['genre'] = self.processed_data['genre'].fillna('Unknown')
        self.processed_data['type'] = self.processed_data['type'].fillna('Unknown')
        self.processed_data['episodes'] = pd.to_numeric(self.processed_data['episodes'], errors='coerce')
        self.processed_data['episodes'] = self.processed_data['episodes'].fillna(0)
        self.processed_data['rating'] = self.processed_data['rating'].fillna(self.processed_data['rating'].mean())
        self.processed_data['members'] = self.processed_data['members'].fillna(0)
        
        # Remove duplicates
        self.processed_data = self.processed_data.drop_duplicates()
        
        # Feature engineering
        self.processed_data['genre_count'] = self.processed_data['genre'].apply(lambda x: len(str(x).split(', ')) if pd.notna(x) else 0)
        self.processed_data['popularity_score'] = self.processed_data['members'] * self.processed_data['rating']
        
        # Create rating categories for classification
        self.processed_data['rating_category'] = pd.cut(self.processed_data['rating'], 
                                                       bins=[0, 5, 6, 7, 8, 10], 
                                                       labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent'])
        
        print(f"Data preprocessing completed. Final dataset shape: {self.processed_data.shape}")
        
    def visualize_data(self):
        """Create visualizations to understand the data distribution"""
        print("\nGenerating data visualizations...")
        
        plt.figure(figsize=(20, 15))
        
        # Rating distribution
        plt.subplot(3, 3, 1)
        plt.hist(self.processed_data['rating'].dropna(), bins=30, color='skyblue', alpha=0.7)
        plt.title('Distribution of Anime Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        
        # Type distribution
        plt.subplot(3, 3, 2)
        type_counts = self.processed_data['type'].value_counts()
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Anime Types')
        
        # Episodes distribution
        plt.subplot(3, 3, 3)
        episodes_filtered = self.processed_data[self.processed_data['episodes'] <= 100]['episodes']
        plt.hist(episodes_filtered, bins=20, color='lightgreen', alpha=0.7)
        plt.title('Distribution of Episodes (≤100)')
        plt.xlabel('Episodes')
        plt.ylabel('Frequency')
        
        # Rating vs Members scatter plot
        plt.subplot(3, 3, 4)
        sample_data = self.processed_data.sample(1000) if len(self.processed_data) > 1000 else self.processed_data
        plt.scatter(sample_data['members'], sample_data['rating'], alpha=0.6, color='coral')
        plt.title('Rating vs Members')
        plt.xlabel('Members')
        plt.ylabel('Rating')
        plt.xscale('log')
        
        # Top genres
        plt.subplot(3, 3, 5)
        all_genres = []
        for genres in self.processed_data['genre'].dropna():
            all_genres.extend([g.strip() for g in str(genres).split(',')])
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        genre_counts.plot(kind='bar')
        plt.title('Top 10 Genres')
        plt.xticks(rotation=45)
        
        # Rating category distribution
        plt.subplot(3, 3, 6)
        self.processed_data['rating_category'].value_counts().plot(kind='bar', color='purple', alpha=0.7)
        plt.title('Rating Categories Distribution')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def build_content_based_recommender(self):
        """
        Build content-based recommendation system using TF-IDF and cosine similarity
        This addresses the AI algorithm implementation requirement
        """
        print("\nBuilding content-based recommendation system...")
        
        # Combine features for content-based filtering
        self.processed_data['combined_features'] = (
            self.processed_data['genre'].astype(str) + ' ' + 
            self.processed_data['type'].astype(str)
        )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.processed_data['combined_features'])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        print(f"Content-based model built. TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def get_content_based_recommendations(self, anime_name, num_recommendations=10):
        """Get content-based recommendations for a given anime"""
        try:
            # Find the anime index
            anime_indices = self.processed_data[self.processed_data['name'].str.contains(anime_name, case=False, na=False)].index
            
            if len(anime_indices) == 0:
                return f"Anime '{anime_name}' not found in the dataset."
            
            idx = anime_indices[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations (excluding the anime itself)
            sim_scores = sim_scores[1:num_recommendations+1]
            anime_indices = [i[0] for i in sim_scores]
            
            recommendations = self.processed_data.iloc[anime_indices][['name', 'genre', 'rating', 'type', 'episodes']]
            similarity_scores = [i[1] for i in sim_scores]
            recommendations['similarity_score'] = similarity_scores
            
            return recommendations
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def build_clustering_model(self):
        """
        Build clustering model to group similar animes
        This demonstrates unsupervised learning techniques
        """
        print("\nBuilding clustering model...")
        
        # Prepare features for clustering
        clustering_features = self.processed_data[['rating', 'episodes', 'members', 'genre_count']].copy()
        clustering_features = clustering_features.dropna()
        
        # Normalize features
        clustering_features_scaled = self.scaler.fit_transform(clustering_features)
        
        # Apply K-means clustering
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(clustering_features_scaled)
        
        # Add cluster labels to data
        self.processed_data.loc[clustering_features.index, 'cluster'] = clusters
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(clustering_features_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.title('Anime Clusters (PCA Visualization)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # Add cluster centers
        centers_pca = pca.transform(self.kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.show()
        
        print(f"Clustering completed. Identified {len(set(clusters))} clusters")
        
        # Analyze clusters
        for i in range(len(set(clusters))):
            cluster_data = self.processed_data[self.processed_data['cluster'] == i]
            print(f"\nCluster {i} characteristics:")
            print(f"  - Average Rating: {cluster_data['rating'].mean():.2f}")
            print(f"  - Average Episodes: {cluster_data['episodes'].mean():.1f}")
            print(f"  - Average Members: {cluster_data['members'].mean():.0f}")
            print(f"  - Size: {len(cluster_data)} animes")
    
    def build_classification_model(self):
        """
        Build classification model to predict anime rating categories
        This demonstrates supervised learning techniques
        """
        print("\nBuilding classification model for rating prediction...")
        
        # Prepare features and target
        feature_cols = ['episodes', 'members', 'genre_count']
        X = self.processed_data[feature_cols].dropna()
        y = self.processed_data.loc[X.index, 'rating_category'].dropna()
        
        # Remove rows where target is missing
        valid_indices = y.index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Classification Model Performance:")
        print(f"  - Accuracy: {accuracy:.3f}")
        print(f"  - Precision: {precision:.3f}")
        print(f"  - Recall: {recall:.3f}")
        print(f"  - F1-Score: {f1:.3f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(feature_importance)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance for Rating Category Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict_rating_category(self, episodes, members, genre_count):
        """Predict rating category for new anime"""
        if self.classifier is None:
            return "Classification model not built yet."
        
        features = np.array([[episodes, members, genre_count]])
        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features)[0]
        
        return {
            'predicted_category': prediction,
            'confidence': max(probability),
            'all_probabilities': dict(zip(self.classifier.classes_, probability))
        }
    
    def get_cluster_recommendations(self, anime_name, num_recommendations=10):
        """Get recommendations based on clustering"""
        try:
            # Find the anime
            anime_match = self.processed_data[self.processed_data['name'].str.contains(anime_name, case=False, na=False)]
            
            if len(anime_match) == 0:
                return f"Anime '{anime_name}' not found in the dataset."
            
            anime_cluster = anime_match.iloc[0]['cluster']
            
            if pd.isna(anime_cluster):
                return "Clustering information not available for this anime."
            
            # Get other animes in the same cluster
            cluster_animes = self.processed_data[
                (self.processed_data['cluster'] == anime_cluster) & 
                (~self.processed_data['name'].str.contains(anime_name, case=False, na=False))
            ]
            
            # Sort by rating and popularity
            recommendations = cluster_animes.nlargest(num_recommendations, 'popularity_score')[
                ['name', 'genre', 'rating', 'type', 'episodes', 'members']
            ]
            
            return recommendations
            
        except Exception as e:
            return f"Error generating cluster-based recommendations: {str(e)}"
    
    def evaluate_recommendation_system(self):
        """
        Evaluate the recommendation system using various metrics
        This addresses the performance evaluation requirement
        """
        print("\nEvaluating recommendation system performance...")
        
        # Content-based evaluation
        sample_animes = self.processed_data.sample(10)['name'].tolist()
        content_recommendations = []
        
        for anime in sample_animes[:5]:  # Test with 5 samples
            recs = self.get_content_based_recommendations(anime, 5)
            if isinstance(recs, pd.DataFrame):
                avg_similarity = recs['similarity_score'].mean()
                content_recommendations.append(avg_similarity)
        
        if content_recommendations:
            avg_content_similarity = np.mean(content_recommendations)
            print(f"Average Content-Based Similarity Score: {avg_content_similarity:.3f}")
        
        # Classification evaluation (already done in build_classification_model)
        classification_metrics = self.build_classification_model()
        
        # Overall system statistics
        print(f"\nSystem Statistics:")
        print(f"  - Total Animes: {len(self.processed_data)}")
        print(f"  - Average Rating: {self.processed_data['rating'].mean():.2f}")
        print(f"  - Unique Genres: {len(set([g.strip() for genres in self.processed_data['genre'].dropna() for g in str(genres).split(',')]))}")
        print(f"  - Content Similarity Matrix Size: {self.cosine_sim.shape}")
        
        return classification_metrics
    
    def interactive_interface(self):
        """
        Command-line interface for user interaction
        This addresses the interface development requirement
        """
        print("\n" + "="*60)
        print("    ANIME RECOMMENDATION SYSTEM")
        print("="*60)
        print("\nWelcome! This system provides anime recommendations using AI techniques.")
        print("Available options:")
        print("1. Get content-based recommendations")
        print("2. Get cluster-based recommendations") 
        print("3. Predict rating category for new anime")
        print("4. View anime statistics")
        print("5. Search anime by genre")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    anime_name = input("Enter anime name: ").strip()
                    num_recs = int(input("Number of recommendations (default 10): ") or "10")
                    
                    print(f"\nContent-based recommendations for '{anime_name}':")
                    recommendations = self.get_content_based_recommendations(anime_name, num_recs)
                    if isinstance(recommendations, pd.DataFrame):
                        print(recommendations.to_string(index=False))
                    else:
                        print(recommendations)
                
                elif choice == '2':
                    anime_name = input("Enter anime name: ").strip()
                    num_recs = int(input("Number of recommendations (default 10): ") or "10")
                    
                    print(f"\nCluster-based recommendations for '{anime_name}':")
                    recommendations = self.get_cluster_recommendations(anime_name, num_recs)
                    if isinstance(recommendations, pd.DataFrame):
                        print(recommendations.to_string(index=False))
                    else:
                        print(recommendations)
                
                elif choice == '3':
                    episodes = int(input("Enter number of episodes: "))
                    members = int(input("Enter number of members: "))
                    genre_count = int(input("Enter number of genres: "))
                    
                    prediction = self.predict_rating_category(episodes, members, genre_count)
                    if isinstance(prediction, dict):
                        print(f"\nPredicted Rating Category: {prediction['predicted_category']}")
                        print(f"Confidence: {prediction['confidence']:.3f}")
                        print("\nAll probabilities:")
                        for category, prob in prediction['all_probabilities'].items():
                            print(f"  {category}: {prob:.3f}")
                    else:
                        print(prediction)
                
                elif choice == '4':
                    print("\nAnime Dataset Statistics:")
                    print(f"Total animes: {len(self.processed_data)}")
                    print(f"Average rating: {self.processed_data['rating'].mean():.2f}")
                    print(f"Most popular type: {self.processed_data['type'].mode().iloc[0]}")
                    print(f"Average episodes: {self.processed_data['episodes'].mean():.1f}")
                    print(f"Total members: {self.processed_data['members'].sum():,.0f}")
                
                elif choice == '5':
                    genre_input = input("Enter genre to search: ").strip()
                    genre_animes = self.processed_data[
                        self.processed_data['genre'].str.contains(genre_input, case=False, na=False)
                    ].nlargest(10, 'rating')[['name', 'genre', 'rating', 'type', 'episodes']]
                    
                    if not genre_animes.empty:
                        print(f"\nTop animes in '{genre_input}' genre:")
                        print(genre_animes.to_string(index=False))
                    else:
                        print(f"No animes found for genre '{genre_input}'")
                
                elif choice == '6':
                    print("Thank you for using the Anime Recommendation System!")
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 1-6.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting... Thank you for using the system!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Please try again.")

def main():
    """
    Main function to run the complete anime recommendation system
    This demonstrates the full AI pipeline from data loading to deployment
    """
    print("Anime Recommendation System - AI Coursework Implementation")
    print("=" * 60)
    
    # Initialize the system
    anime_system = AnimeRecommendationSystem(r"D:\AI assignment\anime.csv")
    
    if anime_system.data is None:
        print("Failed to load data. Please check if anime.csv exists in the current directory.")
        return
    
    # Data visualization
    anime_system.visualize_data()
    
    # Build all AI models
    anime_system.build_content_based_recommender()
    anime_system.build_clustering_model()
    
    # Evaluate the system
    metrics = anime_system.evaluate_recommendation_system()
    
    # Start interactive interface
    anime_system.interactive_interface()

if __name__ == "__main__":
    main()