# Anime Recommendation System - Data Training Module
# Separate module focused on training and validating ML models
# Author: AI Developer | Modular Training Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                   cross_val_score, StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix,
                           silhouette_score, calinski_harabasz_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class AnimeDataTrainer:
    """
    Specialized class for training anime recommendation models.
    Handles data preprocessing, model training, validation, and model persistence.
    """
    
    def __init__(self, data_path, model_save_dir="trained_models"):
        """
        Initialize the training module.
        
        Args:
            data_path (str): Path to the anime dataset
            model_save_dir (str): Directory to save trained models
        """
        self.data_path = data_path
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.feature_matrix = None
        
        # Model containers
        self.trained_models = {}
        self.model_performance = {}
        self.best_models = {}
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        self.label_encoders = {}
        
        # Training metadata
        self.training_info = {
            'training_date': None,
            'dataset_size': None,
            'feature_count': None,
            'models_trained': []
        }
        
        print(f"🔧 Training module initialized")
        print(f"📁 Models will be saved to: {self.model_save_dir}")
    
    def load_and_prepare_data(self):
        """
        Load and prepare data for training with comprehensive preprocessing.
        """
        print("\n📊 LOADING AND PREPARING DATA")
        print("=" * 50)
        
        try:
            # Load dataset
            self.raw_data = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {len(self.raw_data):,} records")
            
            # Display dataset info
            self._display_dataset_summary()
            
            # Process data
            self.processed_data = self._preprocess_data()
            self.feature_matrix = self._engineer_features()
            
            # Update training info
            self.training_info.update({
                'dataset_size': len(self.processed_data),
                'feature_count': self.feature_matrix.shape[1] if self.feature_matrix is not None else 0
            })
            
            print(f"✅ Data preparation completed")
            print(f"📊 Final dataset shape: {self.processed_data.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def _display_dataset_summary(self):
        """Display comprehensive dataset summary."""
        print(f"\n📋 Dataset Summary:")
        print(f"   Shape: {self.raw_data.shape}")
        print(f"   Memory Usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing_data = self.raw_data.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n❗ Missing Values:")
            for col, count in missing_data[missing_data > 0].items():
                percentage = (count / len(self.raw_data)) * 100
                print(f"   {col}: {count:,} ({percentage:.1f}%)")
        
        # Basic statistics
        print(f"\n📈 Key Statistics:")
        if 'rating' in self.raw_data.columns:
            print(f"   Rating range: {self.raw_data['rating'].min():.1f} - {self.raw_data['rating'].max():.1f}")
            print(f"   Average rating: {self.raw_data['rating'].mean():.2f}")
        
        if 'episodes' in self.raw_data.columns:
            print(f"   Episode range: {self.raw_data['episodes'].min()} - {self.raw_data['episodes'].max()}")
        
        if 'type' in self.raw_data.columns:
            print(f"   Most common type: {self.raw_data['type'].mode().iloc[0]}")
    
    def _preprocess_data(self):
        """
        Comprehensive data preprocessing pipeline.
        """
        print("\n🧹 Preprocessing data...")
        
        data = self.raw_data.copy()
        
        # Handle missing values
        data['genre'] = data['genre'].fillna('Unknown')
        data['type'] = data['type'].fillna('TV')
        
        # Convert episodes to numeric
        data['episodes'] = pd.to_numeric(data['episodes'], errors='coerce')
        data['episodes'] = data['episodes'].fillna(1)
        
        # Handle rating missing values
        data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
        data = self._impute_ratings(data)
        
        # Handle members
        data['members'] = pd.to_numeric(data['members'], errors='coerce')
        data['members'] = data['members'].fillna(data['members'].median())
        
        # Remove duplicates and invalid records
        initial_count = len(data)
        data = data.drop_duplicates(subset=['name'] if 'name' in data.columns else None)
        data = data[data['rating'] > 0]
        
        # Handle outliers
        data = self._handle_outliers(data)
        
        print(f"   Removed {initial_count - len(data):,} invalid records")
        print(f"   Final clean dataset: {len(data):,} records")
        
        return data
    
    def _impute_ratings(self, data):
        """Intelligent rating imputation using genre and type similarity."""
        missing_mask = data['rating'].isna()
        
        if missing_mask.sum() > 0:
            print(f"   Imputing {missing_mask.sum()} missing ratings...")
            
            for idx in data[missing_mask].index:
                # Find similar animes
                similar_mask = (
                    (data['genre'] == data.loc[idx, 'genre']) &
                    (data['type'] == data.loc[idx, 'type']) &
                    (~data['rating'].isna())
                )
                
                similar_animes = data[similar_mask]
                
                if len(similar_animes) > 0:
                    # Weighted average by member count
                    if similar_animes['members'].sum() > 0:
                        weights = similar_animes['members'] / similar_animes['members'].sum()
                        imputed_rating = (similar_animes['rating'] * weights).sum()
                    else:
                        imputed_rating = similar_animes['rating'].mean()
                    
                    data.loc[idx, 'rating'] = imputed_rating
                else:
                    data.loc[idx, 'rating'] = data['rating'].median()
        
        return data
    
    def _handle_outliers(self, data):
        """Handle outliers using IQR method."""
        numerical_cols = ['rating', 'episodes', 'members']
        
        for col in numerical_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def _engineer_features(self):
        """
        Comprehensive feature engineering for ML models.
        """
        print("\n⚙️  Engineering features...")
        
        data = self.processed_data.copy()
        
        # Basic derived features
        data['genre_count'] = data['genre'].apply(
            lambda x: len(str(x).split(', ')) if pd.notna(x) else 0
        )
        
        # Logarithmic transformations
        data['log_members'] = np.log1p(data['members'])
        data['log_episodes'] = np.log1p(data['episodes'])
        
        # Popularity and engagement metrics
        data['popularity_score'] = data['members'] * data['rating']
        data['rating_members_ratio'] = data['rating'] / data['log_members']
        
        # Categorical features
        data['is_movie'] = (data['type'] == 'Movie').astype(int)
        data['is_tv'] = (data['type'] == 'TV').astype(int)
        data['is_ova'] = (data['type'] == 'OVA').astype(int)
        
        # Episode-based features
        data['is_long_series'] = (data['episodes'] > 24).astype(int)
        data['is_short_series'] = (data['episodes'] <= 12).astype(int)
        data['is_standard_series'] = ((data['episodes'] > 12) & (data['episodes'] <= 24)).astype(int)
        
        # Rating categories
        data['high_rated'] = (data['rating'] >= 8.0).astype(int)
        data['medium_rated'] = ((data['rating'] >= 6.5) & (data['rating'] < 8.0)).astype(int)
        data['low_rated'] = (data['rating'] < 6.5).astype(int)
        
        # Genre one-hot encoding for popular genres
        popular_genres = ['Comedy', 'Action', 'Drama', 'Romance', 'Fantasy', 
                         'School', 'Supernatural', 'Adventure', 'Sci-Fi', 'Slice of Life']
        
        for genre in popular_genres:
            data[f'genre_{genre.lower().replace(" ", "_")}'] = data['genre'].apply(
                lambda x: 1 if genre in str(x) else 0
            )
        
        # Interaction features
        data['rating_episode_interaction'] = data['rating'] * data['log_episodes']
        data['popularity_rating_interaction'] = data['log_members'] * data['rating']
        
        # Create target variables
        data['rating_category'] = pd.cut(
            data['rating'],
            bins=[0, 5.5, 6.5, 7.5, 8.5, 10],
            labels=['Poor', 'Below_Average', 'Average', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        data['popularity_tier'] = pd.qcut(
            data['members'],
            q=5,
            labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        )
        
        # Update processed data
        self.processed_data = data
        
        # Create feature matrix for ML
        feature_columns = [col for col in data.columns if col not in 
                          ['name', 'genre', 'rating_category', 'popularity_tier']]
        
        feature_matrix = data[feature_columns].select_dtypes(include=[np.number])
        
        print(f"   Created {len(feature_columns)} total features")
        print(f"   Numerical features for ML: {feature_matrix.shape[1]}")
        
        return feature_matrix
    
    def train_content_based_model(self):
        """
        Train content-based recommendation model using TF-IDF.
        """
        print("\n🎯 TRAINING CONTENT-BASED MODEL")
        print("=" * 50)
        
        # Create enhanced content features
        content_features = (
            self.processed_data['genre'].astype(str) + ' ' +
            self.processed_data['type'].astype(str) + ' ' +
            pd.cut(self.processed_data['episodes'],
                  bins=[0, 1, 12, 24, 50, float('inf')],
                  labels=['movie', 'short', 'standard', 'long', 'very_long']).astype(str)
        )
        
        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_features)
        
        # Compute similarity matrix (sample for large datasets)
        if len(self.processed_data) > 5000:
            print("   Large dataset detected, using sample for similarity computation...")
            sample_size = 5000
            sample_indices = np.random.choice(len(self.processed_data), sample_size, replace=False)
            tfidf_sample = tfidf_matrix[sample_indices]
        else:
            tfidf_sample = tfidf_matrix
            sample_indices = np.arange(len(self.processed_data))
        
        cosine_sim = cosine_similarity(tfidf_sample, dense_output=False)
        
        # Store model components
        content_model = {
            'vectorizer': self.tfidf_vectorizer,
            'similarity_matrix': cosine_sim,
            'sample_indices': sample_indices,
            'feature_names': self.tfidf_vectorizer.get_feature_names_out()
        }
        
        self.trained_models['content_based'] = content_model
        
        print(f"✅ Content-based model trained")
        print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"   Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        print(f"   Similarity matrix shape: {cosine_sim.shape}")
        
        return content_model
    
    def train_clustering_model(self):
        """
        Train clustering model with optimal cluster selection.
        """
        print("\n🎪 TRAINING CLUSTERING MODEL")
        print("=" * 50)
        
        # Prepare clustering features
        clustering_features = [
            'rating', 'log_episodes', 'log_members', 'genre_count',
            'popularity_score', 'is_movie', 'is_long_series', 'high_rated'
        ]
        
        X = self.feature_matrix[clustering_features].dropna()
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal number of clusters
        print("   Finding optimal number of clusters...")
        k_range = range(3, 11)
        silhouette_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        # Select optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        print(f"   Optimal clusters: {optimal_k} (Silhouette: {best_silhouette:.3f})")
        
        # Train final model
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(X_scaled)
        
        # Calculate additional metrics
        calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
        
        # Store model
        clustering_model = {
            'model': final_kmeans,
            'scaler': self.scaler,
            'feature_columns': clustering_features,
            'n_clusters': optimal_k,
            'silhouette_score': best_silhouette,
            'calinski_score': calinski_score,
            'cluster_labels': cluster_labels
        }
        
        self.trained_models['clustering'] = clustering_model
        self.model_performance['clustering'] = {
            'silhouette_score': best_silhouette,
            'calinski_harabasz_score': calinski_score,
            'n_clusters': optimal_k
        }
        
        # Add cluster labels to processed data
        self.processed_data.loc[X.index, 'cluster'] = cluster_labels
        
        print(f"✅ Clustering model trained")
        print(f"   Silhouette Score: {best_silhouette:.3f}")
        print(f"   Calinski-Harabasz Score: {calinski_score:.2f}")
        
        return clustering_model
    
    def train_classification_models(self):
        """
        Train and compare multiple classification models with hyperparameter tuning.
        """
        print("\n🤖 TRAINING CLASSIFICATION MODELS")
        print("=" * 50)
        
        # Prepare data for classification
        target_col = 'rating_category'
        feature_cols = [
            'log_episodes', 'log_members', 'genre_count', 'popularity_score',
            'is_movie', 'is_tv', 'is_long_series', 'is_short_series',
            'rating_members_ratio', 'genre_comedy', 'genre_action',
            'genre_drama', 'genre_romance', 'genre_fantasy'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in self.feature_matrix.columns]
        
        X = self.feature_matrix[available_features].dropna()
        y = self.processed_data.loc[X.index, target_col].dropna()
        
        # Remove any remaining NaN values
        valid_indices = y.dropna().index.intersection(X.index)
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        print(f"   Training data shape: {X.shape}")
        print(f"   Target distribution:")
        for category, count in y.value_counts().items():
            print(f"     {category}: {count} ({count/len(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models and hyperparameters
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2']
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        # Train and evaluate models
        classification_results = {}
        best_score = 0
        best_model_name = None
        
        for name, config in models_config.items():
            print(f"\n   Training {name}...")
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate model
            train_score = grid_search.best_score_
            test_predictions = best_model.predict(X_test)
            test_score = f1_score(y_test, test_predictions, average='weighted')
            
            # Additional metrics
            accuracy = accuracy_score(y_test, test_predictions)
            precision = precision_score(y_test, test_predictions, average='weighted')
            recall = recall_score(y_test, test_predictions, average='weighted')
            
            # Store results
            classification_results[name] = {
                'model': best_model,
                'cv_f1_score': train_score,
                'test_f1_score': test_score,
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'best_params': grid_search.best_params_,
                'feature_columns': available_features
            }
            
            print(f"     CV F1: {train_score:.3f}")
            print(f"     Test F1: {test_score:.3f}")
            print(f"     Test Accuracy: {accuracy:.3f}")
            
            # Track best model
            if test_score > best_score:
                best_score = test_score
                best_model_name = name
        
        print(f"\n🏆 Best Model: {best_model_name} (F1: {best_score:.3f})")
        
        # Store models and results
        self.trained_models['classification'] = classification_results
        self.best_models['classification'] = classification_results[best_model_name]
        self.model_performance['classification'] = {
            model: {
                'test_f1_score': results['test_f1_score'],
                'test_accuracy': results['test_accuracy']
            }
            for model, results in classification_results.items()
        }
        
        # Generate detailed evaluation for best model
        self._detailed_classification_evaluation(
            self.best_models['classification'], X_test, y_test
        )
        
        return classification_results
    
    def _detailed_classification_evaluation(self, best_model_info, X_test, y_test):
        """Generate detailed evaluation for the best classification model."""
        print(f"\n📊 Detailed Evaluation - {best_model_info}")
        
        model = best_model_info['model']
        y_pred = model.predict(X_test)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=model.classes_,
                   yticklabels=model.classes_)
        plt.title('Confusion Matrix - Best Classification Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': best_model_info['feature_columns'],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 10 Feature Importance:")
            print(feature_importance.head(10).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance - Best Classification Model')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
    
    def save_models(self):
        """
        Save all trained models and components to disk.
        """
        print("\n💾 SAVING TRAINED MODELS")
        print("=" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_info['training_date'] = timestamp
        self.training_info['models_trained'] = list(self.trained_models.keys())
        
        saved_files = []
        
        try:
            # Save individual models
            for model_name, model_data in self.trained_models.items():
                filename = self.model_save_dir / f"{model_name}_model_{timestamp}.pkl"
                
                with open(filename, 'wb') as f:
                    pickle.dump(model_data, f)
                
                saved_files.append(filename)
                print(f"   ✅ {model_name} model saved: {filename}")
            
            # Save model performance metrics
            performance_file = self.model_save_dir / f"model_performance_{timestamp}.json"
            import json
            with open(performance_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_compatible_performance = {}
                for model, metrics in self.model_performance.items():
                    json_compatible_performance[model] = {}
                    for metric, value in metrics.items():
                        if isinstance(value, (np.integer, np.floating)):
                            json_compatible_performance[model][metric] = float(value)
                        else:
                            json_compatible_performance[model][metric] = value
                
                json.dump(json_compatible_performance, f, indent=2)
            
            saved_files.append(performance_file)
            print(f"   ✅ Performance metrics saved: {performance_file}")
            
            # Save training info
            info_file = self.model_save_dir / f"training_info_{timestamp}.json"
            with open(info_file, 'w') as f:
                json.dump(self.training_info, f, indent=2)
            
            saved_files.append(info_file)
            print(f"   ✅ Training info saved: {info_file}")
            
            # Save processed data sample
            data_sample_file = self.model_save_dir / f"data_sample_{timestamp}.csv"
            sample_size = min(1000, len(self.processed_data))
            self.processed_data.sample(sample_size).to_csv(data_sample_file, index=False)
            saved_files.append(data_sample_file)
            print(f"   ✅ Data sample saved: {data_sample_file}")
            
            print(f"\n🎉 All models saved successfully!")
            print(f"📁 Total files saved: {len(saved_files)}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return []
    
    def generate_training_report(self):
        """
        Generate comprehensive training report.
        """
        print("\n📊 COMPREHENSIVE TRAINING REPORT")
        print("=" * 70)
        
        # Dataset summary
        print(f"\n📋 DATASET SUMMARY")
        print(f"   Total Records: {len(self.processed_data):,}")
        print(f"   Features Engineered: {self.feature_matrix.shape[1] if self.feature_matrix is not None else 0}")
        print(f"   Memory Usage: {self.processed_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Model performance summary
        if self.model_performance:
            print(f"\n🏆 MODEL PERFORMANCE SUMMARY")
            
            if 'classification' in self.model_performance:
                print(f"\n   📊 Classification Models:")
                for model, metrics in self.model_performance['classification'].items():
                    print(f"     {model}:")
                    print(f"       F1-Score: {metrics['test_f1_score']:.3f}")
                    print(f"       Accuracy: {metrics['test_accuracy']:.3f}")
            
            if 'clustering' in self.model_performance:
                print(f"\n   🎪 Clustering Model:")
                metrics = self.model_performance['clustering']
                print(f"     Clusters: {metrics['n_clusters']}")
                print(f"     Silhouette Score: {metrics['silhouette_score']:.3f}")
                print(f"     Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        
        # Training recommendations
        print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
        self._generate_improvement_recommendations()
        
        print(f"\n✅ Training completed successfully!")
        print(f"🕒 Training session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _generate_improvement_recommendations(self):
        """Generate recommendations for model improvement."""
        recommendations = []
        
        # Check classification performance
        if 'classification' in self.model_performance:
            best_f1 = max([metrics['test_f1_score'] for metrics in self.model_performance['classification'].values()])
            
            if best_f1 < 0.7:
                recommendations.append("• Consider collecting more training data")
                recommendations.append("• Try deep learning models (neural networks)")
                recommendations.append("• Add more sophisticated features (embeddings, external data)")
            elif best_f1 < 0.8:
                recommendations.append("• Fine-tune hyperparameters further")
                recommendations.append("• Try ensemble methods with voting")
                recommendations.append("• Consider feature selection techniques")
        
        # Check clustering performance
        if 'clustering' in self.model_performance:
            silhouette = self.model_performance['clustering']['silhouette_score']
            
            if silhouette < 0.3:
                recommendations.append("• Try different clustering algorithms (DBSCAN, Hierarchical)")
                recommendations.append("• Use dimensionality reduction before clustering (PCA, t-SNE)")
                recommendations.append("• Experiment with different distance metrics")
        
        # General recommendations
        recommendations.extend([
            "• Implement cross-validation with more folds",
            "• Try automated feature engineering tools",
            "• Consider using pre-trained embeddings for text data",
            "• Implement model stacking/blending techniques",
            "• Add more domain-specific features (studios, years, seasons)"
        ])
        
        # Display recommendations
        for rec in recommendations[:8]:  # Show top 8
            print(f"   {rec}")

def main():
    """
    Main function to demonstrate the training pipeline.
    """
    print("🚀 ANIME RECOMMENDATION SYSTEM - TRAINING MODULE")
    print("🤖 Comprehensive ML Training Pipeline")
    print("=" * 70)
    
    # Configuration
    DATA_PATH = "anime_cleaned.csv"  # Update this path
    MODEL_SAVE_DIR = "trained_models"
    
    try:
        # Initialize trainer
        trainer = AnimeDataTrainer(DATA_PATH, MODEL_SAVE_DIR)
        
        # Load and prepare data
        trainer.load_and_prepare_data()
        
        # Train all models
        print("\n🔧 STARTING MODEL TRAINING PIPELINE")
        print("=" * 50)
        
        # 1. Content-based model
        content_model = trainer.train_content_based_model()
        
        # 2. Clustering model
        clustering_model = trainer.train_clustering_model()
        
        # 3. Classification models
        classification_models = trainer.train_classification_models()
        
        # Save all models
        saved_files = trainer.save_models()
        
        # Generate comprehensive report
        trainer.generate_training_report()
        
        # Interactive model testing
        while True:
            print("\n" + "=" * 50)
            print("🧪 MODEL TESTING OPTIONS")
            print("1. Test best classification model")
            print("2. View model performance comparison")
            print("3. Generate sample predictions")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                test_classification_interactively(trainer)
            elif choice == '2':
                display_model_comparison(trainer)
            elif choice == '3':
                generate_sample_predictions(trainer)
            elif choice == '4':
                print("\n🎌 Training session completed!")
                print("💫 Models are ready for deployment!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file '{DATA_PATH}'")
        print("Please ensure the anime dataset CSV file exists and update the path.")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("Please check your data and try again.")

def test_classification_interactively(trainer):
    """Interactive testing of the best classification model."""
    if 'classification' not in trainer.best_models:
        print("❌ No classification model available for testing.")
        return
    
    print("\n🧪 INTERACTIVE CLASSIFICATION TESTING")
    print("=" * 50)
    
    best_model_info = trainer.best_models['classification']
    model = best_model_info['model']
    feature_columns = best_model_info['feature_columns']
    
    print("📝 Enter anime characteristics for prediction:")
    
    try:
        # Get user input
        episodes = float(input("Episodes: "))
        members = float(input("Members: "))
        genre_count = float(input("Number of genres: "))
        
        # Create feature vector
        log_episodes = np.log1p(episodes)
        log_members = np.log1p(members)
        popularity_score = members * 7.0  # Assume average rating
        
        # Basic features
        features = {
            'log_episodes': log_episodes,
            'log_members': log_members,
            'genre_count': genre_count,
            'popularity_score': popularity_score,
            'is_movie': 1 if episodes == 1 else 0,
            'is_tv': 1 if episodes > 1 else 0,
            'is_long_series': 1 if episodes > 24 else 0,
            'is_short_series': 1 if episodes <= 12 else 0,
            'rating_members_ratio': 7.0 / log_members if log_members > 0 else 0
        }
        
        # Add genre features (default to 0)
        genre_features = ['genre_comedy', 'genre_action', 'genre_drama', 
                         'genre_romance', 'genre_fantasy']
        for genre_feat in genre_features:
            features[genre_feat] = 0
        
        # Create feature vector in correct order
        feature_vector = []
        for col in feature_columns:
            feature_vector.append(features.get(col, 0))
        
        feature_array = np.array([feature_vector])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        confidence = max(probabilities)
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Predicted Category: {prediction}")
        print(f"   Confidence: {confidence:.1%}")
        
        print(f"\n📊 All Category Probabilities:")
        for category, prob in zip(model.classes_, probabilities):
            print(f"   {category}: {prob:.1%}")
        
    except ValueError:
        print("❌ Please enter valid numeric values.")
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")

def display_model_comparison(trainer):
    """Display detailed model performance comparison."""
    print("\n📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    
    if 'classification' in trainer.model_performance:
        print("\n🤖 CLASSIFICATION MODELS:")
        print(f"{'Model':<20} {'F1-Score':<10} {'Accuracy':<10} {'Status':<10}")
        print("-" * 50)
        
        best_f1 = 0
        best_model = ""
        
        for model_name, metrics in trainer.model_performance['classification'].items():
            f1_score = metrics['test_f1_score']
            accuracy = metrics['test_accuracy']
            status = "⭐ BEST" if f1_score > best_f1 else ""
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_name
            
            print(f"{model_name:<20} {f1_score:<10.3f} {accuracy:<10.3f} {status:<10}")
    
    if 'clustering' in trainer.model_performance:
        print("\n🎪 CLUSTERING MODEL:")
        metrics = trainer.model_performance['clustering']
        print(f"   Clusters: {metrics['n_clusters']}")
        print(f"   Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"   Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        
        # Quality assessment
        silhouette = metrics['silhouette_score']
        if silhouette > 0.5:
            quality = "Excellent"
        elif silhouette > 0.3:
            quality = "Good"
        elif silhouette > 0.1:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"   Quality Assessment: {quality}")

def generate_sample_predictions(trainer):
    """Generate predictions for sample data."""
    if 'classification' not in trainer.best_models:
        print("❌ No classification model available.")
        return
    
    print("\n🎲 SAMPLE PREDICTIONS")
    print("=" * 50)
    
    # Create sample anime scenarios
    sample_scenarios = [
        {
            'name': 'Popular Long Series',
            'episodes': 500,
            'members': 500000,
            'genre_count': 4,
            'description': 'Long-running shounen anime'
        },
        {
            'name': 'Anime Movie',
            'episodes': 1,
            'members': 100000,
            'genre_count': 2,
            'description': 'Studio Ghibli-style movie'
        },
        {
            'name': 'Short Series',
            'episodes': 12,
            'members': 50000,
            'genre_count': 3,
            'description': 'Seasonal slice-of-life anime'
        },
        {
            'name': 'Niche OVA',
            'episodes': 6,
            'members': 5000,
            'genre_count': 2,
            'description': 'Specialized niche content'
        }
    ]
    
    best_model_info = trainer.best_models['classification']
    model = best_model_info['model']
    feature_columns = best_model_info['feature_columns']
    
    for scenario in sample_scenarios:
        print(f"\n🎯 {scenario['name']} ({scenario['description']})")
        
        # Create features
        episodes = scenario['episodes']
        members = scenario['members']
        genre_count = scenario['genre_count']
        
        log_episodes = np.log1p(episodes)
        log_members = np.log1p(members)
        popularity_score = members * 7.0
        
        features = {
            'log_episodes': log_episodes,
            'log_members': log_members,
            'genre_count': genre_count,
            'popularity_score': popularity_score,
            'is_movie': 1 if episodes == 1 else 0,
            'is_tv': 1 if episodes > 1 else 0,
            'is_long_series': 1 if episodes > 24 else 0,
            'is_short_series': 1 if episodes <= 12 else 0,
            'rating_members_ratio': 7.0 / log_members if log_members > 0 else 0,
            'genre_comedy': 0,
            'genre_action': 1 if 'Long Series' in scenario['name'] else 0,
            'genre_drama': 1 if 'Movie' in scenario['name'] else 0,
            'genre_romance': 1 if 'Short Series' in scenario['name'] else 0,
            'genre_fantasy': 0
        }
        
        # Create feature vector
        feature_vector = [features.get(col, 0) for col in feature_columns]
        feature_array = np.array([feature_vector])
        
        try:
            prediction = model.predict(feature_array)[0]
            probabilities = model.predict_proba(feature_array)[0]
            confidence = max(probabilities)
            
            print(f"   Prediction: {prediction} (Confidence: {confidence:.1%})")
            
        except Exception as e:
            print(f"   Error: {str(e)}")

# Training pipeline class for batch processing
class BatchTrainingPipeline:
    """
    Batch training pipeline for processing multiple datasets or configurations.
    """
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.training_results = []
    
    def run_training_experiments(self, experiments):
        """
        Run multiple training experiments with different configurations.
        
        Args:
            experiments (list): List of experiment configurations
        """
        print("🔬 RUNNING TRAINING EXPERIMENTS")
        print("=" * 50)
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n🧪 Experiment {i}/{len(experiments)}: {experiment.get('name', 'Unnamed')}")
            
            try:
                # Merge base config with experiment config
                config = {**self.base_config, **experiment}
                
                # Initialize trainer
                trainer = AnimeDataTrainer(
                    config['data_path'],
                    config.get('model_save_dir', f"experiment_{i}_models")
                )
                
                # Run training pipeline
                trainer.load_and_prepare_data()
                
                if config.get('train_content', True):
                    trainer.train_content_based_model()
                
                if config.get('train_clustering', True):
                    trainer.train_clustering_model()
                
                if config.get('train_classification', True):
                    trainer.train_classification_models()
                
                # Save results
                if config.get('save_models', True):
                    trainer.save_models()
                
                # Store results
                self.training_results.append({
                    'experiment': experiment.get('name', f'Experiment_{i}'),
                    'performance': trainer.model_performance,
                    'config': config
                })
                
                print(f"✅ Experiment {i} completed successfully")
                
            except Exception as e:
                print(f"❌ Experiment {i} failed: {str(e)}")
                self.training_results.append({
                    'experiment': experiment.get('name', f'Experiment_{i}'),
                    'error': str(e),
                    'config': config
                })
    
    def compare_experiments(self):
        """Compare results across all experiments."""
        print("\n📊 EXPERIMENT COMPARISON")
        print("=" * 70)
        
        if not self.training_results:
            print("❌ No experiment results to compare.")
            return
        
        # Compare classification results
        print("\n🤖 Classification Model Comparison:")
        print(f"{'Experiment':<20} {'Best F1':<10} {'Best Model':<20}")
        print("-" * 50)
        
        for result in self.training_results:
            if 'performance' in result and 'classification' in result['performance']:
                classification_perf = result['performance']['classification']
                best_f1 = max([metrics['test_f1_score'] for metrics in classification_perf.values()])
                best_model = max(classification_perf.keys(), 
                               key=lambda k: classification_perf[k]['test_f1_score'])
                
                print(f"{result['experiment']:<20} {best_f1:<10.3f} {best_model:<20}")
            else:
                print(f"{result['experiment']:<20} {'ERROR':<10} {'N/A':<20}")

if __name__ == "__main__":
    main()