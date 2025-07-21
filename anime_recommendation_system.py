# Enhanced Anime Recommendation System with Improved Accuracy
# Author: AI Developer | Enhanced for better performance and maintainability

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnimeRecommendationSystem:
    """
    Enhanced Anime Recommendation System with improved accuracy and comprehensive AI techniques.
    
    Features implemented:
    1. Content-based filtering using advanced NLP (TF-IDF + N-grams)
    2. Hybrid recommendation approach combining multiple algorithms
    3. Advanced feature engineering for better model performance
    4. Ensemble methods for classification with hyperparameter tuning
    5. Cross-validation and model selection
    6. Improved clustering with optimal cluster selection
    7. Comprehensive evaluation metrics and validation
    """
    
    def __init__(self, file_path):
        """
        Initialize the enhanced recommendation system.
        
        Args:
            file_path (str): Path to the anime dataset CSV file
        """
        # Core data structures
        self.data = None                    # Original dataset
        self.processed_data = None          # Cleaned and processed dataset
        
        # Content-based filtering components
        self.tfidf_vectorizer = None        # TF-IDF vectorizer for text features
        self.tfidf_matrix = None           # TF-IDF feature matrix
        self.cosine_sim = None             # Cosine similarity matrix
        
        # Preprocessing and scaling
        self.scaler = StandardScaler()      # Feature scaler for numerical data
        self.label_encoders = {}            # Dictionary to store label encoders
        
        # Machine learning models
        self.kmeans = None                  # K-means clustering model
        self.best_classifier = None         # Best performing classification model
        self.model_pipeline = None          # Complete ML pipeline
        
        # Model performance tracking
        self.model_scores = {}              # Dictionary to store model performance
        
        # Initialize the system
        self.load_and_preprocess_data(file_path)
    
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the anime dataset with enhanced data cleaning.
        Implements robust data handling with outlier detection and advanced feature engineering.
        
        Args:
            file_path (str): Path to the CSV file
        """
        print("🔄 Loading and preprocessing data...")
        
        try:
            # Load dataset with error handling
            self.data = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully: {len(self.data):,} records")
        except FileNotFoundError:
            print("❌ Error: anime.csv file not found. Please check the file path.")
            return
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return
        
        # Display comprehensive dataset information
        self._display_dataset_info()
        
        # Create a copy for processing
        self.processed_data = self.data.copy()
        
        # Enhanced data cleaning pipeline
        self._clean_data()
        self._engineer_features()
        self._handle_outliers()
        self._create_target_variables()
        
        print(f"✅ Data preprocessing completed. Final shape: {self.processed_data.shape}")
        print(f"📊 Dataset quality score: {self._calculate_data_quality_score():.2f}/10")
    
    def _display_dataset_info(self):
        """Display comprehensive information about the dataset."""
        print(f"\n📋 Dataset Information:")
        print(f"   Shape: {self.data.shape}")
        print(f"   Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\n🔍 Missing Values Analysis:")
        missing_info = self.data.isnull().sum()
        for col, missing in missing_info[missing_info > 0].items():
            percentage = (missing / len(self.data)) * 100
            print(f"   {col}: {missing:,} ({percentage:.1f}%)")
        
        print(f"\n📊 Data Types:")
        print(self.data.dtypes.value_counts())
    
    def _clean_data(self):
        """
        Enhanced data cleaning with intelligent missing value handling.
        Uses domain knowledge to fill missing values appropriately.
        """
        print("🧹 Performing enhanced data cleaning...")
        
        # Handle missing values with domain-specific logic
        self.processed_data['genre'] = self.processed_data['genre'].fillna('Unknown')
        self.processed_data['type'] = self.processed_data['type'].fillna('TV')  # Most common type
        
        # Convert episodes to numeric, handle special cases
        self.processed_data['episodes'] = pd.to_numeric(
            self.processed_data['episodes'], errors='coerce'
        )
        
        # Fill missing episodes based on anime type (domain knowledge)
        type_episode_median = self.processed_data.groupby('type')['episodes'].median()
        for anime_type, median_episodes in type_episode_median.items():
            mask = (self.processed_data['type'] == anime_type) & \
                   (self.processed_data['episodes'].isna())
            self.processed_data.loc[mask, 'episodes'] = median_episodes
        
        # Handle remaining missing episodes
        self.processed_data['episodes'] = self.processed_data['episodes'].fillna(1)
        
        # Smart rating imputation using similar animes
        self._impute_ratings_intelligently()
        
        # Handle members with median by type
        self.processed_data['members'] = self.processed_data['members'].fillna(
            self.processed_data['members'].median()
        )
        
        # Remove duplicates and invalid records
        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data.drop_duplicates(subset=['name'])
        self.processed_data = self.processed_data[self.processed_data['rating'] > 0]
        
        print(f"   Removed {initial_count - len(self.processed_data):,} duplicate/invalid records")
    
    def _impute_ratings_intelligently(self):
        """
        Intelligent rating imputation using content-based similarity.
        For animes with missing ratings, predict based on similar animes.
        """
        missing_rating_mask = self.processed_data['rating'].isna()
        
        if missing_rating_mask.sum() > 0:
            print(f"   Imputing {missing_rating_mask.sum()} missing ratings intelligently...")
            
            # Use genre similarity for rating imputation
            for idx in self.processed_data[missing_rating_mask].index:
                similar_animes = self.processed_data[
                    (self.processed_data['genre'] == self.processed_data.loc[idx, 'genre']) &
                    (self.processed_data['type'] == self.processed_data.loc[idx, 'type']) &
                    (~self.processed_data['rating'].isna())
                ]
                
                if len(similar_animes) > 0:
                    # Use weighted average based on member count
                    weights = similar_animes['members'] / similar_animes['members'].sum()
                    imputed_rating = (similar_animes['rating'] * weights).sum()
                    self.processed_data.loc[idx, 'rating'] = imputed_rating
                else:
                    # Fallback to overall median
                    self.processed_data.loc[idx, 'rating'] = self.processed_data['rating'].median()
    
    def _engineer_features(self):
        """
        Advanced feature engineering to improve model performance.
        Creates multiple derived features that capture anime characteristics.
        """
        print("⚙️  Engineering advanced features...")
        
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
        
        # Genre-based features (one-hot encoding for popular genres)
        popular_genres = ['Comedy', 'Action', 'Drama', 'Romance', 'Fantasy', 'School', 'Supernatural']
        for genre in popular_genres:
            self.processed_data[f'genre_{genre.lower()}'] = self.processed_data['genre'].apply(
                lambda x: 1 if genre in str(x) else 0
            )
        
        # Interaction features
        self.processed_data['rating_episode_interaction'] = (
            self.processed_data['rating'] * np.log1p(self.processed_data['episodes'])
        )
        
        print(f"   Created {len(self.processed_data.columns) - len(self.data.columns)} new features")
    
    def _handle_outliers(self):
        """
        Detect and handle outliers using statistical methods.
        Uses IQR method to cap extreme values rather than removing them.
        """
        print("🎯 Handling outliers...")
        
        numerical_columns = ['rating', 'episodes', 'members']
        outlier_count = 0
        
        for col in numerical_columns:
            Q1 = self.processed_data[col].quantile(0.25)
            Q3 = self.processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            outliers_mask = (
                (self.processed_data[col] < lower_bound) | 
                (self.processed_data[col] > upper_bound)
            )
            outlier_count += outliers_mask.sum()
            
            # Cap the values
            self.processed_data[col] = self.processed_data[col].clip(
                lower=max(lower_bound, self.processed_data[col].min()),
                upper=upper_bound
            )
        
        print(f"   Capped {outlier_count} outlier values")
    
    def _create_target_variables(self):
        """
        Create multiple target variables for different prediction tasks.
        """
        # Enhanced rating categories with more granular classification
        self.processed_data['rating_category'] = pd.cut(
            self.processed_data['rating'], 
            bins=[0, 5.5, 6.5, 7.5, 8.5, 10], 
            labels=['Poor', 'Below_Average', 'Average', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        # Binary classification for high-quality anime
        self.processed_data['is_highly_rated'] = (self.processed_data['rating'] >= 8.0).astype(int)
        
        # Popularity categories
        self.processed_data['popularity_category'] = pd.qcut(
            self.processed_data['members'], 
            q=5, 
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        )
    
    def _calculate_data_quality_score(self):
        """
        Calculate a data quality score based on completeness and consistency.
        
        Returns:
            float: Quality score from 0-10
        """
        # Completeness score (percentage of non-null values)
        completeness = (1 - self.processed_data.isnull().sum().sum() / 
                       (len(self.processed_data) * len(self.processed_data.columns)))
        
        # Consistency score (no negative ratings, reasonable episode counts)
        consistency_checks = [
            (self.processed_data['rating'] >= 0).all(),
            (self.processed_data['episodes'] >= 0).all(),
            (self.processed_data['members'] >= 0).all(),
        ]
        consistency = sum(consistency_checks) / len(consistency_checks)
        
        return (completeness * 0.7 + consistency * 0.3) * 10
    
    def build_enhanced_content_recommender(self):
        """
        Build an enhanced content-based recommendation system.
        Uses advanced NLP techniques including n-grams and improved feature weighting.
        """
        print("🔧 Building enhanced content-based recommendation system...")
        
        # Create comprehensive feature text combining multiple attributes
        self.processed_data['enhanced_features'] = (
            self.processed_data['genre'].astype(str) + ' ' + 
            self.processed_data['type'].astype(str) + ' ' +
            # Add episode range as categorical feature
            pd.cut(self.processed_data['episodes'], 
                  bins=[0, 1, 12, 24, 50, float('inf')], 
                  labels=['Movie', 'Short', 'Standard', 'Long', 'Very_Long']).astype(str)
        )
        
        # Enhanced TF-IDF with n-grams and optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=8000,          # Increased feature space
            ngram_range=(1, 2),         # Include bigrams for better context
            min_df=2,                   # Ignore very rare terms
            max_df=0.95,                # Ignore very common terms
            sublinear_tf=True,          # Apply sublinear scaling
        )
        
        # Fit and transform the enhanced features
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.processed_data['enhanced_features']
        )
        
        # Calculate cosine similarity with optimized computation
        print("   Computing cosine similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, dense_output=False)
        
        print(f"✅ Enhanced content-based model built")
        print(f"   TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"   Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def get_hybrid_recommendations(self, anime_name, num_recommendations=10, 
                                 content_weight=0.7, rating_weight=0.3):
        """
        Get hybrid recommendations combining content similarity and rating-based filtering.
        
        Args:
            anime_name (str): Name of the anime to get recommendations for
            num_recommendations (int): Number of recommendations to return
            content_weight (float): Weight for content-based similarity
            rating_weight (float): Weight for rating-based scoring
            
        Returns:
            pd.DataFrame or str: Recommendations or error message
        """
        try:
            # Find the anime with fuzzy matching
            anime_matches = self.processed_data[
                self.processed_data['name'].str.contains(anime_name, case=False, na=False)
            ]
            
            if len(anime_matches) == 0:
                return f"❌ Anime '{anime_name}' not found. Try a different name."
            
            # Use the best match (first result)
            idx = anime_matches.index[0]
            matched_anime = anime_matches.iloc[0]
            
            print(f"🎯 Found: '{matched_anime['name']}' (Rating: {matched_anime['rating']:.2f})")
            
            # Get content-based similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx].toarray()[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get candidate recommendations (excluding the input anime)
            candidate_indices = [i[0] for i in sim_scores[1:num_recommendations*3]]  # Get more candidates
            candidates = self.processed_data.iloc[candidate_indices].copy()
            
            # Add similarity scores
            candidates['content_similarity'] = [sim_scores[i+1][1] for i in range(len(candidates))]
            
            # Calculate hybrid score combining content similarity and rating
            candidates['rating_score'] = (candidates['rating'] - candidates['rating'].min()) / \
                                       (candidates['rating'].max() - candidates['rating'].min())
            
            candidates['hybrid_score'] = (
                content_weight * candidates['content_similarity'] + 
                rating_weight * candidates['rating_score']
            )
            
            # Sort by hybrid score and get top recommendations
            recommendations = candidates.nlargest(num_recommendations, 'hybrid_score')[
                ['name', 'genre', 'rating', 'type', 'episodes', 'members', 
                 'content_similarity', 'hybrid_score']
            ].round({
                'rating': 2, 
                'content_similarity': 3, 
                'hybrid_score': 3
            })
            
            return recommendations
            
        except Exception as e:
            return f"❌ Error generating recommendations: {str(e)}"
    
    def build_optimal_clustering_model(self):
        """
        Build clustering model with optimal number of clusters using elbow method and silhouette analysis.
        """
        print("🔧 Building optimal clustering model...")
        
        # Prepare enhanced features for clustering
        clustering_features = [
            'rating', 'log_episodes', 'log_members', 'genre_count', 
            'popularity_score', 'is_movie', 'is_long_series', 'high_rated'
        ]
        
        X_cluster = self.processed_data[clustering_features].copy()
        X_cluster = X_cluster.dropna()
        
        # Scale features for clustering
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        print("   Finding optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        
        # Select optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"   Optimal number of clusters: {optimal_k}")
        print(f"   Best silhouette score: {max(silhouette_scores):.3f}")
        
        # Build final clustering model
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        self.processed_data.loc[X_cluster.index, 'cluster'] = clusters
        
        # Visualize clusters
        self._visualize_clusters(X_scaled, clusters, optimal_k)
        self._analyze_clusters(optimal_k)
        
        print(f"✅ Clustering completed with {optimal_k} clusters")
    
    def _visualize_clusters(self, X_scaled, clusters, n_clusters):
        """Visualize clusters using PCA."""
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=clusters, cmap='viridis', alpha=0.6)
        plt.title(f'Anime Clusters Visualization (n_clusters={n_clusters})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, label='Cluster')
        
        # Add cluster centers
        centers_pca = pca.transform(self.kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='x', s=300, linewidths=3, label='Centroids')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _analyze_clusters(self, n_clusters):
        """Analyze and describe each cluster."""
        print("\n📊 Cluster Analysis:")
        
        for i in range(n_clusters):
            cluster_data = self.processed_data[self.processed_data['cluster'] == i]
            
            print(f"\n🎯 Cluster {i} ({len(cluster_data)} animes):")
            print(f"   Average Rating: {cluster_data['rating'].mean():.2f}")
            print(f"   Average Episodes: {cluster_data['episodes'].mean():.1f}")
            print(f"   Average Members: {cluster_data['members'].mean():,.0f}")
            print(f"   Most Common Type: {cluster_data['type'].mode().iloc[0]}")
            
            # Top genres in this cluster
            all_genres = []
            for genres in cluster_data['genre'].dropna():
                all_genres.extend([g.strip() for g in str(genres).split(',')])
            
            if all_genres:
                top_genres = pd.Series(all_genres).value_counts().head(3)
                print(f"   Top Genres: {', '.join(top_genres.index.tolist())}")
    
    def build_ensemble_classifier(self):
        """
        Build an ensemble classification model with hyperparameter tuning.
        Uses multiple algorithms and selects the best performer.
        """
        print("🚀 Building ensemble classification model with hyperparameter tuning...")
        
        # Prepare enhanced feature set
        feature_columns = [
            'episodes', 'log_members', 'genre_count', 'popularity_score',
            'is_movie', 'is_long_series', 'is_short_series', 'rating_members_ratio',
            'genre_comedy', 'genre_action', 'genre_drama', 'genre_romance'
        ]
        
        # Prepare data
        X = self.processed_data[feature_columns].dropna()
        y = self.processed_data.loc[X.index, 'rating_category'].dropna()
        
        # Remove any remaining NaN values
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models and their hyperparameters for tuning
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
        
        # Train and evaluate each model with cross-validation
        best_score = 0
        best_model_name = None
        
        print("\n🔍 Model comparison with cross-validation:")
        
        for name, config in models.items():
            print(f"\n   Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Best model from grid search
            best_model = grid_search.best_estimator_
            
            # Cross-validation score
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=5, scoring='f1_weighted'
            )
            
            # Test score
            test_predictions = best_model.predict(X_test)
            test_score = f1_score(y_test, test_predictions, average='weighted')
            
            # Store results
            self.model_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_f1': test_score,
                'best_params': grid_search.best_params_
            }
            
            print(f"     CV F1: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
            print(f"     Test F1: {test_score:.3f}")
            print(f"     Best params: {grid_search.best_params_}")
            
            # Track best model
            if test_score > best_score:
                best_score = test_score
                best_model_name = name
                self.best_classifier = best_model
        
        print(f"\n🏆 Best Model: {best_model_name} (F1: {best_score:.3f})")
        
        # Detailed evaluation of best model
        self._evaluate_best_model(X_test, y_test, feature_columns)
        
        return self.model_scores
    
    def _evaluate_best_model(self, X_test, y_test, feature_columns):
        """
        Comprehensive evaluation of the best performing model.
        """
        print(f"\n📊 Detailed evaluation of best model:")
        
        # Predictions
        y_pred = self.best_classifier.predict(X_test)
        
        # Calculate all metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Display metrics
        for metric, score in metrics.items():
            print(f"   {metric}: {score:.3f}")
        
        # Feature importance (if available)
        if hasattr(self.best_classifier, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.best_classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 5 Most Important Features:")
            for _, row in feature_importance.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
            
            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            top_features = feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def predict_rating_category_enhanced(self, episodes, members, genre_count, 
                                       anime_type='TV', genres=''):
        """
        Enhanced prediction with more features and confidence intervals.
        
        Args:
            episodes (int): Number of episodes
            members (int): Number of members
            genre_count (int): Number of genres
            anime_type (str): Type of anime (TV, Movie, etc.)
            genres (str): Comma-separated genres
            
        Returns:
            dict: Prediction results with confidence and explanations
        """
        if self.best_classifier is None:
            return "❌ Classification model not trained yet."
        
        # Engineer features similar to training data
        log_members = np.log1p(members)
        popularity_score = members * 7.0  # Assume average rating for new anime
        is_movie = 1 if anime_type.lower() == 'movie' else 0
        is_long_series = 1 if episodes > 24 else 0
        is_short_series = 1 if episodes <= 12 else 0
        rating_members_ratio = 7.0 / log_members if log_members > 0 else 0
        
        # Genre features
        genre_features = {}
        popular_genres = ['comedy', 'action', 'drama', 'romance']
        genres_lower = genres.lower()
        
        for genre in popular_genres:
            genre_features[f'genre_{genre}'] = 1 if genre in genres_lower else 0
        
        # Create feature vector
        features = np.array([[
            episodes, log_members, genre_count, popularity_score,
            is_movie, is_long_series, is_short_series, rating_members_ratio,
            genre_features['genre_comedy'], genre_features['genre_action'],
            genre_features['genre_drama'], genre_features['genre_romance']
        ]])
        
        # Make prediction
        prediction = self.best_classifier.predict(features)[0]
        probabilities = self.best_classifier.predict_proba(features)[0]
        
        # Calculate confidence and uncertainty
        max_prob = max(probabilities)
        uncertainty = 1 - max_prob
        
        # Generate explanation
        explanation = self._generate_prediction_explanation(
            episodes, members, genre_count, anime_type, max_prob
        )
        
        return {
            'predicted_category': prediction,
            'confidence': max_prob,
            'uncertainty': uncertainty,
            'all_probabilities': dict(zip(self.best_classifier.classes_, probabilities)),
            'explanation': explanation,
            'recommendation': self._get_category_recommendation(prediction, max_prob)
        }
    
    def _generate_prediction_explanation(self, episodes, members, genre_count, anime_type, confidence):
        """Generate human-readable explanation for the prediction."""
        explanations = []
        
        # Episode-based insights
        if episodes == 1:
            explanations.append("Single episode suggests a movie or special")
        elif episodes <= 12:
            explanations.append("Short series format typically indicates focused storytelling")
        elif episodes > 50:
            explanations.append("Long series suggests extensive world-building")
        
        # Member-based insights
        if members < 1000:
            explanations.append("Low member count suggests niche appeal")
        elif members > 100000:
            explanations.append("High member count indicates mainstream popularity")
        
        # Confidence insight
        if confidence > 0.8:
            explanations.append("High prediction confidence based on strong feature patterns")
        elif confidence < 0.6:
            explanations.append("Moderate confidence - anime characteristics are somewhat ambiguous")
        
        return " | ".join(explanations)
    
    def _get_category_recommendation(self, category, confidence):
        """Provide actionable recommendations based on prediction."""
        recommendations = {
            'Excellent': "This anime is predicted to be exceptional! Definitely worth watching.",
            'Good': "Solid choice with good ratings expected. Recommended for most viewers.",
            'Average': "Standard anime that meets basic expectations. Good for casual viewing.",
            'Below_Average': "May appeal to specific niches but generally below average quality.",
            'Poor': "Consider alternatives unless you're specifically interested in this type."
        }
        
        base_rec = recommendations.get(category, "Prediction uncertain.")
        
        if confidence < 0.7:
            base_rec += " (Note: Prediction has moderate uncertainty - consider additional reviews)"
        
        return base_rec
    
    def get_smart_cluster_recommendations(self, anime_name, num_recommendations=10):
        """
        Enhanced cluster-based recommendations with similarity ranking.
        """
        try:
            # Find the anime with fuzzy matching
            anime_matches = self.processed_data[
                self.processed_data['name'].str.contains(anime_name, case=False, na=False)
            ]
            
            if len(anime_matches) == 0:
                return f"❌ Anime '{anime_name}' not found in the dataset."
            
            anime_data = anime_matches.iloc[0]
            anime_cluster = anime_data['cluster']
            
            if pd.isna(anime_cluster):
                return "❌ Clustering information not available for this anime."
            
            print(f"🎯 Found: '{anime_data['name']}' in Cluster {int(anime_cluster)}")
            
            # Get all animes in the same cluster
            cluster_animes = self.processed_data[
                (self.processed_data['cluster'] == anime_cluster) & 
                (~self.processed_data['name'].str.contains(anime_name, case=False, na=False))
            ].copy()
            
            if len(cluster_animes) == 0:
                return "❌ No other animes found in the same cluster."
            
            # Calculate similarity score based on multiple factors
            reference_rating = anime_data['rating']
            reference_episodes = anime_data['episodes']
            reference_genre_count = anime_data['genre_count']
            
            # Multi-factor similarity scoring
            cluster_animes['rating_similarity'] = 1 - abs(cluster_animes['rating'] - reference_rating) / 10
            cluster_animes['episode_similarity'] = np.exp(-abs(np.log1p(cluster_animes['episodes']) - 
                                                             np.log1p(reference_episodes)) / 2)
            cluster_animes['genre_similarity'] = 1 - abs(cluster_animes['genre_count'] - 
                                                        reference_genre_count) / max(cluster_animes['genre_count'].max(), 1)
            
            # Combined similarity score
            cluster_animes['cluster_similarity'] = (
                0.4 * cluster_animes['rating_similarity'] +
                0.3 * cluster_animes['episode_similarity'] +
                0.3 * cluster_animes['genre_similarity']
            )
            
            # Sort by similarity and rating
            recommendations = cluster_animes.nlargest(
                num_recommendations, ['cluster_similarity', 'rating']
            )[['name', 'genre', 'rating', 'type', 'episodes', 'members', 'cluster_similarity']]
            
            recommendations['cluster_similarity'] = recommendations['cluster_similarity'].round(3)
            
            return recommendations
            
        except Exception as e:
            return f"❌ Error generating cluster recommendations: {str(e)}"
    
    def comprehensive_evaluation(self):
        """
        Comprehensive evaluation of the entire recommendation system.
        Tests all components and provides detailed performance metrics.
        """
        print("\n🔍 COMPREHENSIVE SYSTEM EVALUATION")
        print("=" * 50)
        
        evaluation_results = {}
        
        # 1. Content-based recommendation evaluation
        print("\n📊 1. Content-Based Recommendation Quality:")
        content_scores = []
        sample_animes = self.processed_data.sample(min(20, len(self.processed_data)))
        
        for _, anime in sample_animes.iterrows():
            recs = self.get_hybrid_recommendations(anime['name'], 5)
            if isinstance(recs, pd.DataFrame) and len(recs) > 0:
                avg_similarity = recs['content_similarity'].mean()
                content_scores.append(avg_similarity)
        
        if content_scores:
            evaluation_results['content_based'] = {
                'avg_similarity': np.mean(content_scores),
                'std_similarity': np.std(content_scores),
                'coverage': len(content_scores) / len(sample_animes)
            }
            
            print(f"   Average Content Similarity: {np.mean(content_scores):.3f}")
            print(f"   Similarity Std Dev: {np.std(content_scores):.3f}")
            print(f"   Recommendation Coverage: {len(content_scores)/len(sample_animes)*100:.1f}%")
        
        # 2. Classification model evaluation (already done in build_ensemble_classifier)
        if hasattr(self, 'model_scores'):
            print(f"\n📊 2. Classification Model Performance:")
            evaluation_results['classification'] = self.model_scores
            
            for model_name, scores in self.model_scores.items():
                print(f"   {model_name}:")
                print(f"     Test F1-Score: {scores['test_f1']:.3f}")
                print(f"     CV F1-Score: {scores['cv_mean']:.3f} (±{scores['cv_std']*2:.3f})")
        
        # 3. Clustering quality evaluation
        if self.kmeans is not None:
            print(f"\n📊 3. Clustering Quality:")
            
            # Prepare clustering features
            clustering_features = [
                'rating', 'log_episodes', 'log_members', 'genre_count', 
                'popularity_score', 'is_movie', 'is_long_series', 'high_rated'
            ]
            
            X_cluster = self.processed_data[clustering_features].dropna()
            X_scaled = self.scaler.fit_transform(X_cluster)
            cluster_labels = self.processed_data.loc[X_cluster.index, 'cluster'].dropna()
            
            if len(cluster_labels) > 0:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score
                
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
                
                evaluation_results['clustering'] = {
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': calinski_score,
                    'n_clusters': len(set(cluster_labels))
                }
                
                print(f"   Silhouette Score: {silhouette_avg:.3f}")
                print(f"   Calinski-Harabasz Score: {calinski_score:.2f}")
                print(f"   Number of Clusters: {len(set(cluster_labels))}")
        
        # 4. System coverage and diversity
        print(f"\n📊 4. System Coverage & Diversity:")
        
        total_animes = len(self.processed_data)
        unique_genres = len(set([g.strip() for genres in self.processed_data['genre'].dropna() 
                               for g in str(genres).split(',') if g.strip()]))
        
        coverage_metrics = {
            'total_animes': total_animes,
            'unique_genres': unique_genres,
            'avg_genres_per_anime': self.processed_data['genre_count'].mean(),
            'rating_range': self.processed_data['rating'].max() - self.processed_data['rating'].min(),
            'episode_range': self.processed_data['episodes'].max() - self.processed_data['episodes'].min()
        }
        
        evaluation_results['coverage'] = coverage_metrics
        
        print(f"   Dataset Size: {total_animes:,} animes")
        print(f"   Genre Diversity: {unique_genres} unique genres")
        print(f"   Avg Genres per Anime: {coverage_metrics['avg_genres_per_anime']:.1f}")
        print(f"   Rating Range: {coverage_metrics['rating_range']:.1f}")
        
        # 5. Overall system score
        overall_scores = []
        
        if 'content_based' in evaluation_results:
            overall_scores.append(evaluation_results['content_based']['avg_similarity'] * 10)
        
        if 'classification' in evaluation_results:
            best_f1 = max([scores['test_f1'] for scores in evaluation_results['classification'].values()])
            overall_scores.append(best_f1 * 10)
        
        if 'clustering' in evaluation_results:
            normalized_silhouette = (evaluation_results['clustering']['silhouette_score'] + 1) / 2 * 10
            overall_scores.append(normalized_silhouette)
        
        if overall_scores:
            overall_score = np.mean(overall_scores)
            evaluation_results['overall_score'] = overall_score
            
            print(f"\n🏆 Overall System Score: {overall_score:.1f}/10")
            
            # Provide recommendations for improvement
            self._provide_improvement_recommendations(evaluation_results)
        
        return evaluation_results
    
    def _provide_improvement_recommendations(self, evaluation_results):
        """Provide specific recommendations for system improvement based on evaluation results."""
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
        
        improvements = []
        
        # Content-based improvements
        if 'content_based' in evaluation_results:
            avg_sim = evaluation_results['content_based']['avg_similarity']
            if avg_sim < 0.3:
                improvements.append("• Enhance content features with more descriptive text (synopsis, themes)")
                improvements.append("• Consider using word embeddings instead of TF-IDF")
        
        # Classification improvements
        if 'classification' in evaluation_results:
            best_f1 = max([scores['test_f1'] for scores in evaluation_results['classification'].values()])
            if best_f1 < 0.8:
                improvements.append("• Add more features like studio, year, season information")
                improvements.append("• Try deep learning models (neural networks)")
                improvements.append("• Implement ensemble voting with multiple models")
        
        # Clustering improvements
        if 'clustering' in evaluation_results:
            silhouette = evaluation_results['clustering']['silhouette_score']
            if silhouette < 0.3:
                improvements.append("• Try different clustering algorithms (DBSCAN, hierarchical)")
                improvements.append("• Use dimensionality reduction before clustering")
        
        # General improvements
        improvements.extend([
            "• Implement user-based collaborative filtering",
            "• Add real-time learning capabilities",
            "• Include user feedback mechanisms",
            "• Implement A/B testing for recommendation strategies"
        ])
        
        for improvement in improvements[:5]:  # Show top 5 recommendations
            print(f"   {improvement}")
    
    def enhanced_interactive_interface(self):
        """
        Enhanced command-line interface with more features and better UX.
        """
        print("\n" + "=" * 70)
        print("    🎌 ENHANCED ANIME RECOMMENDATION SYSTEM 🎌")
        print("=" * 70)
        print("\n🤖 Welcome! This AI-powered system provides intelligent anime recommendations.")
        print("\n📋 Available Features:")
        print("   1. 🎯 Get hybrid recommendations (content + rating based)")
        print("   2. 🔮 Predict rating category for new anime")
        print("   3. 🔍 Search anime by name")
        print("   4. 🚪 Exit")
        
        while True:
            try:
                print("\n" + "-" * 50)
                choice = input("🎮 Enter your choice (1-7): ").strip()
                
                if choice == '1':
                    anime_name = input("📝 Enter anime name: ").strip()
                    if not anime_name:
                        print("❌ Please enter a valid anime name.")
                        continue
                        
                    num_recs = input("🔢 Number of recommendations (default 10): ").strip()
                    num_recs = int(num_recs) if num_recs.isdigit() else 10
                    
                    print(f"\n🔄 Generating hybrid recommendations for '{anime_name}'...")
                    recommendations = self.get_hybrid_recommendations(anime_name, num_recs)
                    
                    if isinstance(recommendations, pd.DataFrame):
                        print(f"\n✨ Top {len(recommendations)} recommendations:")
                        print(recommendations.to_string(index=False))
                        
                        # Show summary statistics
                        avg_rating = recommendations['rating'].mean()
                        print(f"\n📈 Recommendations Summary:")
                        print(f"   Average Rating: {avg_rating:.2f}")
                        print(f"   Rating Range: {recommendations['rating'].min():.1f} - {recommendations['rating'].max():.1f}")
                    else:
                        print(recommendations)
                
                elif choice == '2':
                    print("\n🔮 Anime Rating Category Prediction")
                    print("📝 Please provide the following information:")
                    
                    try:
                        episodes = int(input("   Episodes: "))
                        members = int(input("   Expected members: "))
                        genre_count = int(input("   Number of genres: "))
                        anime_type = input("   Type (TV/Movie/OVA/Special): ").strip() or "TV"
                        genres = input("   Genres (comma-separated): ").strip()
                        
                        prediction = self.predict_rating_category_enhanced(
                            episodes, members, genre_count, anime_type, genres
                        )
                        
                        if isinstance(prediction, dict):
                            print(f"\n🎯 Prediction Results:")
                            print(f"   📊 Predicted Category: {prediction['predicted_category']}")
                            print(f"   🎯 Confidence: {prediction['confidence']:.1%}")
                            print(f"   ❓ Uncertainty: {prediction['uncertainty']:.1%}")
                            print(f"\n💡 Explanation: {prediction['explanation']}")
                            print(f"\n💭 Recommendation: {prediction['recommendation']}")
                            
                            print(f"\n📋 All Category Probabilities:")
                            for category, prob in prediction['all_probabilities'].items():
                                print(f"   {category}: {prob:.1%}")
                        else:
                            print(prediction)
                            
                    except ValueError:
                        print("❌ Please enter valid numeric values.")
                
                # elif choice == '3':
                #     anime_name = input("📝 Enter anime name for cluster recommendations: ").strip()
                #     if not anime_name:
                #         print("❌ Please enter a valid anime name.")
                #         continue
                        
                #     num_recs = input("🔢 Number of recommendations (default 10): ").strip()
                #     num_recs = int(num_recs) if num_recs.isdigit() else 10
                    
                #     print(f"\n🔄 Finding cluster-based recommendations...")
                #     recommendations = self.get_smart_cluster_recommendations(anime_name, num_recs)
                    
                #     if isinstance(recommendations, pd.DataFrame):
                #         print(f"\n🎪 Cluster-based recommendations:")
                #         print(recommendations.to_string(index=False))
                #     else:
                #         print(recommendations)
                
                # elif choice == '4':
                #     print("\n📊 COMPREHENSIVE DATASET STATISTICS")
                #     print("=" * 50)
                    
                #     stats = {
                #         'Total Animes': f"{len(self.processed_data):,}",
                #         'Average Rating': f"{self.processed_data['rating'].mean():.2f}",
                #         'Rating Range': f"{self.processed_data['rating'].min():.1f} - {self.processed_data['rating'].max():.1f}",
                #         'Most Popular Type': self.processed_data['type'].mode().iloc[0],
                #         'Average Episodes': f"{self.processed_data['episodes'].mean():.1f}",
                #         'Total Members': f"{self.processed_data['members'].sum():,}",
                #         'Unique Genres': len(set([g.strip() for genres in self.processed_data['genre'].dropna() 
                #                                 for g in str(genres).split(',') if g.strip()]))
                #     }
                    
                #     for key, value in stats.items():
                #         print(f"   {key}: {value}")
                    
                #     # Top genres
                #     all_genres = []
                #     for genres in self.processed_data['genre'].dropna():
                #         all_genres.extend([g.strip() for g in str(genres).split(',')])
                #     top_genres = pd.Series(all_genres).value_counts().head(5)
                    
                #     print(f"\n🏆 Top 5 Genres:")
                #     for genre, count in top_genres.items():
                #         print(f"   {genre}: {count:,} animes")
                
                elif choice == '3':
                    search_term = input("🔍 Enter search term: ").strip().lower()
                    if not search_term:
                        print("❌ Please enter a search term.")
                        continue
                        
                    matches = self.processed_data[
                        self.processed_data['name'].str.contains(search_term, case=False, na=False)
                    ].head(10)
                    
                    if len(matches) > 0:
                        print(f"\n🎯 Found {len(matches)} matches:")
                        display_cols = ['name', 'rating', 'type', 'episodes', 'genre']
                        print(matches[display_cols].to_string(index=False))
                    else:
                        print("❌ No matches found. Try a different search term.")
                
                # elif choice == '6':
                #     print("\n🔄 Running comprehensive system evaluation...")
                #     print("⏳ This may take a moment...")
                #     evaluation_results = self.comprehensive_evaluation()
                    
                #     if 'overall_score' in evaluation_results:
                #         score = evaluation_results['overall_score']
                #         if score >= 8:
                #             print("🌟 Excellent system performance!")
                #         elif score >= 6:
                #             print("✅ Good system performance with room for improvement.")
                #         else:
                #             print("⚠️  System needs optimization.")
                
                elif choice == '4':
                    print("\n🎌 Thank you for using the Enhanced Anime Recommendation System!")
                    print("💫 May you discover amazing anime! Sayonara! 👋")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter a number between 1-7.")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user. Exiting gracefully...")
                print("👋 Thank you for using the system!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
                print("🔄 Please try again or choose a different option.")

def main():
    """
    Enhanced main function with comprehensive system initialization and execution.
    Demonstrates the complete AI pipeline from data loading to interactive deployment.
    """
    print("🚀 ENHANCED ANIME RECOMMENDATION SYSTEM")
    print("🤖 Implementing Advanced AI & Machine Learning Techniques")
    print("=" * 70)
    
    # Initialize the enhanced system
    print("🔧 Initializing system...")
    anime_system = EnhancedAnimeRecommendationSystem(r"D:\AI assignment\anime_cleaned.csv")
    
    if anime_system.data is None:
        print("❌ System initialization failed. Please check the data file.")
        return
    
    print("✅ System initialized successfully!")
    
    try:
        # Build all AI models with enhanced techniques
        print("\n🔧 Building AI models...")
        
        # 1. Enhanced content-based recommender
        anime_system.build_enhanced_content_recommender()
        
        # 2. Optimal clustering model
        anime_system.build_optimal_clustering_model()
        
        # 3. Ensemble classification model
        print("\n🚀 Training ensemble models...")
        model_scores = anime_system.build_ensemble_classifier()
        
        print(f"\n✅ All AI models built successfully!")
        print(f"🏆 Best classifier performance: {max([scores['test_f1'] for scores in model_scores.values()]):.3f} F1-Score")
        
        # Launch enhanced interactive interface
        anime_system.enhanced_interactive_interface()
        
    except Exception as e:
        print(f"❌ Error during system execution: {str(e)}")
        print("🔍 Please check your data file and try again.")

if __name__ == "__main__":
    main()