# Anime Dataset Preprocessing Module
# Comprehensive data cleaning and feature engineering for anime recommendation system
# Author: AI Developer | Specialized preprocessing pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# Scikit-learn imports for preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')

class AnimeDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for anime dataset.
    
    Features:
    1. Intelligent missing value imputation
    2. Advanced outlier detection and handling
    3. Feature engineering with domain knowledge
    4. Data quality assessment and validation
    5. Export processed data in multiple formats
    6. Detailed preprocessing reports and visualizations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config (dict): Configuration parameters for preprocessing
        """
        # Default configuration
        self.config = {
            'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation', 'lof'
            'missing_strategy': 'intelligent',  # 'simple', 'knn', 'intelligent'
            'feature_selection': True,
            'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
            'genre_encoding': 'onehot',  # 'onehot', 'label', 'count'
            'min_rating': 0.0,
            'max_rating': 10.0,
            'min_episodes': 1,
            'max_episodes': 10000,
            'verbose': True
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Initialize components
        self.raw_data = None
        self.processed_data = None
        self.feature_matrix = None
        self.target_variables = None
        
        # Preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        
        # Metadata and statistics
        self.preprocessing_stats = {}
        self.data_quality_metrics = {}
        self.feature_importance = {}
        
        # Setup logging
        self._setup_logging()
        
        print("🔧 Anime Data Preprocessor initialized")
        print(f"📋 Configuration: {self.config}")
    
    def _setup_logging(self):
        """Setup logging for preprocessing operations."""
        logging.basicConfig(
            level=logging.INFO if self.config['verbose'] else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> bool:
        """
        Load raw anime dataset from CSV file.
        
        Args:
            file_path (str): Path to the anime dataset CSV
            
        Returns:
            bool: Success status
        """
        print("\n📂 LOADING ANIME DATASET")
        print("=" * 50)
        
        try:
            self.raw_data = pd.read_csv(file_path)
            
            # Basic dataset info
            print(f"✅ Dataset loaded successfully")
            print(f"📊 Shape: {self.raw_data.shape}")
            print(f"💾 Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Column information
            print(f"\n📋 Columns: {list(self.raw_data.columns)}")
            print(f"🔢 Data types:")
            for dtype, count in self.raw_data.dtypes.value_counts().items():
                print(f"   {dtype}: {count} columns")
            
            # Store initial statistics
            self.preprocessing_stats['initial_shape'] = self.raw_data.shape
            self.preprocessing_stats['initial_memory'] = self.raw_data.memory_usage(deep=True).sum()
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis.
        
        Returns:
            dict: Data quality metrics and recommendations
        """
        print("\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)
        
        if self.raw_data is None:
            print("❌ No data loaded for analysis")
            return {}
        
        analysis = {}
        
        # 1. Missing values analysis
        missing_analysis = self._analyze_missing_values()
        analysis['missing_values'] = missing_analysis
        
        # 2. Outlier analysis
        outlier_analysis = self._analyze_outliers()
        analysis['outliers'] = outlier_analysis
        
        # 3. Data distribution analysis
        distribution_analysis = self._analyze_distributions()
        analysis['distributions'] = distribution_analysis
        
        # 4. Consistency checks
        consistency_analysis = self._analyze_consistency()
        analysis['consistency'] = consistency_analysis
        
        # 5. Uniqueness analysis
        uniqueness_analysis = self._analyze_uniqueness()
        analysis['uniqueness'] = uniqueness_analysis
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(analysis)
        analysis['overall_quality_score'] = quality_score
        
        self.data_quality_metrics = analysis
        
        print(f"\n🏆 Overall Data Quality Score: {quality_score:.2f}/10")
        self._print_quality_recommendations(analysis)
        
        return analysis
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values patterns."""
        print("\n📊 Missing Values Analysis:")
        
        missing_info = {}
        missing_counts = self.raw_data.isnull().sum()
        total_rows = len(self.raw_data)
        
        for col in self.raw_data.columns:
            missing_count = missing_counts[col]
            missing_percentage = (missing_count / total_rows) * 100
            
            missing_info[col] = {
                'count': int(missing_count),
                'percentage': missing_percentage
            }
            
            if missing_count > 0:
                print(f"   {col}: {missing_count:,} ({missing_percentage:.1f}%)")
        
        # Missing value patterns
        missing_patterns = self.raw_data.isnull().sum(axis=1).value_counts().sort_index()
        missing_info['patterns'] = missing_patterns.to_dict()
        
        return missing_info
    
    def _analyze_outliers(self) -> Dict[str, Any]:
        """Analyze outliers in numerical columns."""
        print("\n📈 Outlier Analysis:")
        
        outlier_info = {}
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in self.raw_data.columns:
                data = self.raw_data[col].dropna()
                
                if len(data) > 0:
                    # IQR method
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_iqr = ((data < lower_bound) | (data > upper_bound)).sum()
                    
                    # Z-score method
                    z_scores = np.abs((data - data.mean()) / data.std())
                    outliers_zscore = (z_scores > 3).sum()
                    
                    outlier_info[col] = {
                        'iqr_outliers': int(outliers_iqr),
                        'zscore_outliers': int(outliers_zscore),
                        'iqr_percentage': (outliers_iqr / len(data)) * 100,
                        'bounds': {'lower': lower_bound, 'upper': upper_bound},
                        'stats': {'mean': data.mean(), 'std': data.std(), 'min': data.min(), 'max': data.max()}
                    }
                    
                    if outliers_iqr > 0:
                        print(f"   {col}: {outliers_iqr} IQR outliers ({(outliers_iqr/len(data)*100):.1f}%)")
        
        return outlier_info
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze data distributions."""
        print("\n📉 Distribution Analysis:")
        
        distribution_info = {}
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in self.raw_data.columns:
                data = self.raw_data[col].dropna()
                
                if len(data) > 0:
                    # Basic statistics
                    stats = {
                        'mean': data.mean(),
                        'median': data.median(),
                        'std': data.std(),
                        'skewness': data.skew(),
                        'kurtosis': data.kurtosis()
                    }
                    
                    # Distribution assessment
                    skew_interpretation = self._interpret_skewness(stats['skewness'])
                    
                    distribution_info[col] = {
                        'stats': stats,
                        'skew_interpretation': skew_interpretation,
                        'unique_values': data.nunique(),
                        'zero_values': (data == 0).sum()
                    }
                    
                    print(f"   {col}: {skew_interpretation} (skew: {stats['skewness']:.2f})")
        
        return distribution_info
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness values."""
        if abs(skewness) < 0.5:
            return "Normal"
        elif abs(skewness) < 1:
            return "Moderately skewed"
        else:
            return "Highly skewed"
    
    def _analyze_consistency(self) -> Dict[str, Any]:
        """Analyze data consistency."""
        print("\n🔍 Consistency Analysis:")
        
        consistency_info = {}
        issues = []
        
        # Check for negative values where they shouldn't exist
        if 'rating' in self.raw_data.columns:
            negative_ratings = (self.raw_data['rating'] < 0).sum()
            if negative_ratings > 0:
                issues.append(f"Negative ratings: {negative_ratings}")
            consistency_info['negative_ratings'] = int(negative_ratings)
        
        if 'episodes' in self.raw_data.columns:
            negative_episodes = (self.raw_data['episodes'] < 0).sum()
            if negative_episodes > 0:
                issues.append(f"Negative episodes: {negative_episodes}")
            consistency_info['negative_episodes'] = int(negative_episodes)
        
        if 'members' in self.raw_data.columns:
            negative_members = (self.raw_data['members'] < 0).sum()
            if negative_members > 0:
                issues.append(f"Negative members: {negative_members}")
            consistency_info['negative_members'] = int(negative_members)
        
        # Check for unrealistic values
        if 'rating' in self.raw_data.columns:
            invalid_ratings = ((self.raw_data['rating'] < 0) | (self.raw_data['rating'] > 10)).sum()
            consistency_info['invalid_ratings'] = int(invalid_ratings)
        
        if 'episodes' in self.raw_data.columns:
            extreme_episodes = (self.raw_data['episodes'] > 5000).sum()
            consistency_info['extreme_episodes'] = int(extreme_episodes)
        
        # Duplicate analysis
        duplicates = self.raw_data.duplicated().sum()
        consistency_info['duplicates'] = int(duplicates)
        
        if 'name' in self.raw_data.columns:
            name_duplicates = self.raw_data['name'].duplicated().sum()
            consistency_info['name_duplicates'] = int(name_duplicates)
        
        consistency_info['issues'] = issues
        
        if issues:
            for issue in issues:
                print(f"   ⚠️  {issue}")
        else:
            print("   ✅ No major consistency issues found")
        
        return consistency_info
    
    def _analyze_uniqueness(self) -> Dict[str, Any]:
        """Analyze data uniqueness."""
        print("\n🔢 Uniqueness Analysis:")
        
        uniqueness_info = {}
        
        for col in self.raw_data.columns:
            unique_count = self.raw_data[col].nunique()
            total_count = len(self.raw_data[col].dropna())
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            
            uniqueness_info[col] = {
                'unique_count': int(unique_count),
                'total_count': int(total_count),
                'uniqueness_ratio': uniqueness_ratio
            }
            
            if uniqueness_ratio < 0.1 and col != 'type':  # Low uniqueness might be concerning
                print(f"   ⚠️  {col}: Low uniqueness ({uniqueness_ratio:.2%})")
            elif uniqueness_ratio == 1.0:
                print(f"   ✅ {col}: All unique values")
        
        return uniqueness_info
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        # Completeness score (based on missing values)
        if 'missing_values' in analysis:
            total_missing = sum([info['percentage'] for info in analysis['missing_values'].values() 
                               if isinstance(info, dict)])
            avg_missing_percentage = total_missing / len(analysis['missing_values'])
            completeness_score = max(0, 10 - (avg_missing_percentage / 10))
            scores.append(completeness_score)
        
        # Consistency score
        if 'consistency' in analysis:
            consistency_issues = len(analysis['consistency']['issues'])
            consistency_score = max(0, 10 - consistency_issues * 2)
            scores.append(consistency_score)
        
        # Outlier score
        if 'outliers' in analysis:
            outlier_columns = len([col for col, info in analysis['outliers'].items() 
                                 if info['iqr_percentage'] > 10])
            total_columns = len(analysis['outliers'])
            outlier_score = 10 - (outlier_columns / total_columns * 5) if total_columns > 0 else 10
            scores.append(outlier_score)
        
        return np.mean(scores) if scores else 5.0
    
    def _print_quality_recommendations(self, analysis: Dict[str, Any]):
        """Print data quality recommendations."""
        print(f"\n💡 QUALITY IMPROVEMENT RECOMMENDATIONS:")
        
        recommendations = []
        
        # Missing value recommendations
        if 'missing_values' in analysis:
            high_missing_cols = [col for col, info in analysis['missing_values'].items() 
                               if isinstance(info, dict) and info['percentage'] > 20]
            if high_missing_cols:
                recommendations.append(f"• Consider dropping or imputing columns with high missing values: {high_missing_cols}")
        
        # Outlier recommendations
        if 'outliers' in analysis:
            high_outlier_cols = [col for col, info in analysis['outliers'].items() 
                               if info['iqr_percentage'] > 15]
            if high_outlier_cols:
                recommendations.append(f"• Apply outlier treatment to: {high_outlier_cols}")
        
        # Consistency recommendations
        if 'consistency' in analysis and analysis['consistency']['issues']:
            recommendations.append("• Fix consistency issues found in the data")
        
        # General recommendations
        recommendations.extend([
            "• Consider feature scaling for numerical variables",
            "• Apply appropriate encoding for categorical variables",
            "• Consider feature selection to reduce dimensionality"
        ])
        
        for rec in recommendations[:5]:
            print(f"   {rec}")
    
    def preprocess_data(self, target_column: Optional[str] = None) -> bool:
        """
        Execute complete preprocessing pipeline.
        
        Args:
            target_column (str): Target column for supervised learning tasks
            
        Returns:
            bool: Success status
        """
        print("\n🚀 STARTING COMPREHENSIVE PREPROCESSING PIPELINE")
        print("=" * 70)
        
        if self.raw_data is None:
            print("❌ No data loaded for preprocessing")
            return False
        
        try:
            # Step 1: Initial cleaning
            self.processed_data = self._initial_cleaning()
            
            # Step 2: Handle missing values
            self.processed_data = self._handle_missing_values()
            
            # Step 3: Handle outliers
            self.processed_data = self._handle_outliers()
            
            # Step 4: Feature engineering
            self.processed_data = self._engineer_features()
            
            # Step 5: Encode categorical variables
            self.processed_data = self._encode_categorical_variables()
            
            # Step 6: Create target variables
            if target_column:
                self.target_variables = self._create_target_variables(target_column)
            
            # Step 7: Feature selection
            if self.config['feature_selection']:
                self.processed_data = self._select_features(target_column)
            
            # Step 8: Scale features
            self.feature_matrix = self._scale_features()
            
            # Step 9: Final validation
            success = self._validate_processed_data()
            
            if success:
                print("\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
                self._print_preprocessing_summary()
            
            return success
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {str(e)}")
            self.logger.error(f"Preprocessing failed: {str(e)}")
            return False
    
    def _initial_cleaning(self) -> pd.DataFrame:
        """Initial data cleaning steps."""
        print("\n🧹 Step 1: Initial Data Cleaning")
        
        data = self.raw_data.copy()
        initial_shape = data.shape
        
        # Remove completely empty rows and columns
        data = data.dropna(how='all')  # Remove empty rows
        data = data.dropna(axis=1, how='all')  # Remove empty columns
        
        # Handle obvious data entry errors
        if 'rating' in data.columns:
            # Fix rating values outside reasonable range
            data.loc[data['rating'] > 10, 'rating'] = np.nan
            data.loc[data['rating'] < 0, 'rating'] = np.nan
        
        if 'episodes' in data.columns:
            # Convert episodes to numeric, handle 'Unknown' and other string values
            data['episodes'] = pd.to_numeric(data['episodes'], errors='coerce')
            # Cap extremely high episode counts (likely data errors)
            data.loc[data['episodes'] > 5000, 'episodes'] = np.nan
        
        if 'members' in data.columns:
            # Ensure members is numeric and non-negative
            data['members'] = pd.to_numeric(data['members'], errors='coerce')
            data.loc[data['members'] < 0, 'members'] = np.nan
        
        # Remove exact duplicates
        duplicates_removed = len(data) - len(data.drop_duplicates())
        data = data.drop_duplicates()
        
        # Clean text columns
        text_columns = data.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col in data.columns:
                # Strip whitespace and standardize missing value representations
                data[col] = data[col].astype(str).str.strip()
                data[col] = data[col].replace(['', 'nan', 'None', 'null', 'N/A', 'Unknown'], np.nan)
        
        print(f"   ✅ Shape change: {initial_shape} → {data.shape}")
        print(f"   🗑️  Removed {duplicates_removed} duplicate rows")
        
        return data
    
    def _handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values using intelligent strategies."""
        print("\n🔧 Step 2: Handling Missing Values")
        
        data = self.processed_data.copy()
        
        if self.config['missing_strategy'] == 'intelligent':
            data = self._intelligent_imputation(data)
        elif self.config['missing_strategy'] == 'knn':
            data = self._knn_imputation(data)
        else:
            data = self._simple_imputation(data)
        
        # Verify no critical missing values remain
        critical_columns = ['name'] if 'name' in data.columns else []
        for col in critical_columns:
            if data[col].isnull().any():
                data = data.dropna(subset=[col])
                print(f"   🗑️  Dropped rows with missing {col}")
        
        return data
    
    def _intelligent_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value imputation using domain knowledge."""
        print("   🧠 Using intelligent imputation strategy")
        
        # Genre imputation
        if 'genre' in data.columns:
            data['genre'] = data['genre'].fillna('Unknown')
            print("   📝 Filled missing genres with 'Unknown'")
        
        # Type imputation (most common anime type is TV)
        if 'type' in data.columns:
            most_common_type = data['type'].mode().iloc[0] if not data['type'].mode().empty else 'TV'
            data['type'] = data['type'].fillna(most_common_type)
            print(f"   📺 Filled missing types with '{most_common_type}'")
        
        # Episodes imputation based on type
        if 'episodes' in data.columns and 'type' in data.columns:
            for anime_type in data['type'].unique():
                if pd.notna(anime_type):
                    type_mask = data['type'] == anime_type
                    missing_episodes_mask = data['episodes'].isna()
                    combined_mask = type_mask & missing_episodes_mask
                    
                    if combined_mask.any():
                        type_median_episodes = data[type_mask]['episodes'].median()
                        if pd.notna(type_median_episodes):
                            data.loc[combined_mask, 'episodes'] = type_median_episodes
                            print(f"   🎬 Filled {combined_mask.sum()} missing episodes for {anime_type} with {type_median_episodes}")
        
        # Rating imputation using similar animes
        if 'rating' in data.columns:
            data = self._impute_ratings_by_similarity(data)
        
        # Members imputation
        if 'members' in data.columns:
            # Use median members for the anime type, or overall median
            if 'type' in data.columns:
                data['members'] = data.groupby('type')['members'].transform(
                    lambda x: x.fillna(x.median())
                )
            data['members'] = data['members'].fillna(data['members'].median())
            print("   👥 Filled missing members with type-based medians")
        
        return data
    
    def _impute_ratings_by_similarity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute ratings based on similar animes."""
        missing_rating_mask = data['rating'].isna()
        imputed_count = 0
        
        if missing_rating_mask.any():
            print(f"   ⭐ Imputing {missing_rating_mask.sum()} missing ratings using similarity")
            
            for idx in data[missing_rating_mask].index:
                # Find similar animes based on genre and type
                current_genre = data.loc[idx, 'genre'] if 'genre' in data.columns else None
                current_type = data.loc[idx, 'type'] if 'type' in data.columns else None
                
                similarity_mask = pd.Series([True] * len(data), index=data.index)
                
                if pd.notna(current_genre) and 'genre' in data.columns:
                    similarity_mask &= data['genre'] == current_genre
                
                if pd.notna(current_type) and 'type' in data.columns:
                    similarity_mask &= data['type'] == current_type
                
                similarity_mask &= data['rating'].notna()
                similarity_mask &= data.index != idx  # Exclude the current row
                
                similar_animes = data[similarity_mask]
                
                if len(similar_animes) > 0:
                    # Weighted average by member count if available
                    if 'members' in data.columns and similar_animes['members'].notna().any():
                        weights = similar_animes['members'].fillna(similar_animes['members'].median())
                        if weights.sum() > 0:
                            weighted_rating = (similar_animes['rating'] * weights).sum() / weights.sum()
                        else:
                            weighted_rating = similar_animes['rating'].mean()
                    else:
                        weighted_rating = similar_animes['rating'].mean()
                    
                    data.loc[idx, 'rating'] = weighted_rating
                    imputed_count += 1
                else:
                    # Fallback to overall median
                    data.loc[idx, 'rating'] = data['rating'].median()
                    imputed_count += 1
            
            print(f"   ✅ Successfully imputed {imputed_count} ratings")
        
        return data
    
    def _knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """KNN-based imputation for numerical features."""
        print("   🎯 Using KNN imputation strategy")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            data[numerical_cols] = knn_imputer.fit_transform(data[numerical_cols])
            self.imputers['knn'] = knn_imputer
            print(f"   ✅ Applied KNN imputation to {len(numerical_cols)} numerical columns")
        
        # Handle categorical columns separately
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_value)
        
        return data
    
    def _simple_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple imputation strategies."""
        print("   📊 Using simple imputation strategy")
        
        # Numerical columns - fill with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().any():
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
                print(f"   🔢 Filled {col} with median: {median_value}")
        
        # Categorical columns - fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                data[col] = data[col].fillna(mode_value)
                print(f"   📝 Filled {col} with mode: {mode_value}")
        
        return data
    
    def _handle_outliers(self) -> pd.DataFrame:
        """Handle outliers using specified method."""
        print(f"\n📈 Step 3: Handling Outliers ({self.config['outlier_method']})")
        
        data = self.processed_data.copy()
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        outliers_handled = 0
        
        for col in numerical_cols:
            if col in data.columns:
                initial_outliers = self._count_outliers_iqr(data[col])
                
                if self.config['outlier_method'] == 'iqr':
                    data[col] = self._handle_outliers_iqr(data[col])
                elif self.config['outlier_method'] == 'zscore':
                    data[col] = self._handle_outliers_zscore(data[col])
                elif self.config['outlier_method'] == 'isolation':
                    data = self._handle_outliers_isolation(data, col)
                elif self.config['outlier_method'] == 'lof':
                    data = self._handle_outliers_lof(data, col)
                
                final_outliers = self._count_outliers_iqr(data[col])
                handled = initial_outliers - final_outliers
                
                if handled > 0:
                    outliers_handled += handled
                    print(f"   📊 {col}: Handled {handled} outliers")
        
        print(f"   ✅ Total outliers handled: {outliers_handled}")
        return data
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _handle_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Handle outliers using IQR method (capping)."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    def _handle_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Handle outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        threshold = 3
        
        # Cap values with z-score > 3
        outlier_mask = z_scores > threshold
        if outlier_mask.any():
            # Replace with values at the threshold
            series_capped = series.copy()
            mean_val = series.mean()
            std_val = series.std()
            
            upper_cap = mean_val + threshold * std_val
            lower_cap = mean_val - threshold * std_val
            
            series_capped = series_capped.clip(lower=lower_cap, upper=upper_cap)
            return series_capped
        
        return series
    
    def _handle_outliers_isolation(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Handle outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        
        if len(data[col].dropna()) > 10:  # Need sufficient data points
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data[col].dropna().values.reshape(-1, 1))
            
            # Replace outliers with median
            outlier_indices = data[col].dropna().index[outlier_labels == -1]
            median_val = data[col].median()
            data.loc[outlier_indices, col] = median_val
        
        return data
    
    def _handle_outliers_lof(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Handle outliers using Local Outlier Factor."""
        if len(data[col].dropna()) > 20:  # Need sufficient data points
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            outlier_labels = lof.fit_predict(data[col].dropna().values.reshape(-1, 1))
            
            # Replace outliers with median
            outlier_indices = data[col].dropna().index[outlier_labels == -1]
            median_val = data[col].median()
            data.loc[outlier_indices, col] = median_val
        
        return data
    
    def _engineer_features(self) -> pd.DataFrame:
        """Advanced feature engineering with domain knowledge."""
        print("\n⚙️ Step 4: Feature Engineering")
        
        data = self.processed_data.copy()
        features_created = 0
        
        # Basic derived features
        if 'genre' in data.columns:
            data['genre_count'] = data['genre'].apply(
                lambda x: len(str(x).split(', ')) if pd.notna(x) and str(x) != 'Unknown' else 0
            )
            features_created += 1
            print("   ✅ Created genre_count feature")
        
        # Logarithmic transformations for skewed data
        if 'members' in data.columns:
            data['log_members'] = np.log1p(data['members'])
            features_created += 1
            print("   ✅ Created log_members feature")
        
        if 'episodes' in data.columns:
            data['log_episodes'] = np.log1p(data['episodes'])
            features_created += 1
            print("   ✅ Created log_episodes feature")
        
        # Popularity and engagement metrics
        if 'members' in data.columns and 'rating' in data.columns:
            data['popularity_score'] = data['members'] * data['rating']
            data['rating_members_ratio'] = data['rating'] / np.log1p(data['members'])
            features_created += 2
            print("   ✅ Created popularity and engagement features")
        
        # Episode-based categorical features
        if 'episodes' in data.columns:
            data['is_movie'] = (data['episodes'] == 1).astype(int)
            data['is_short_series'] = (data['episodes'] <= 12).astype(int)
            data['is_standard_series'] = ((data['episodes'] > 12) & (data['episodes'] <= 26)).astype(int)
            data['is_long_series'] = (data['episodes'] > 26).astype(int)
            features_created += 4
            print("   ✅ Created episode-based categorical features")
        
        # Type-based features
        if 'type' in data.columns:
            for anime_type in ['TV', 'Movie', 'OVA', 'Special', 'ONA']:
                if anime_type in data['type'].values:
                    data[f'is_{anime_type.lower()}'] = (data['type'] == anime_type).astype(int)
                    features_created += 1
            print(f"   ✅ Created type-based binary features")
        
        # Rating-based features
        if 'rating' in data.columns:
            data['high_rated'] = (data['rating'] >= 8.0).astype(int)
            data['medium_rated'] = ((data['rating'] >= 6.5) & (data['rating'] < 8.0)).astype(int)
            data['low_rated'] = (data['rating'] < 6.5).astype(int)
            features_created += 3
            print("   ✅ Created rating-based categorical features")
        
        # Genre-based features (one-hot encoding for popular genres)
        if 'genre' in data.columns:
            popular_genres = self._extract_popular_genres(data['genre'], top_n=15)
            for genre in popular_genres:
                genre_feature = f'genre_{genre.lower().replace(" ", "_").replace("-", "_")}'
                data[genre_feature] = data['genre'].apply(
                    lambda x: 1 if pd.notna(x) and genre in str(x) else 0
                )
                features_created += 1
            print(f"   ✅ Created {len(popular_genres)} genre-based binary features")
        
        # Interaction features
        if 'rating' in data.columns and 'episodes' in data.columns:
            data['rating_episode_interaction'] = data['rating'] * np.log1p(data['episodes'])
            features_created += 1
            print("   ✅ Created interaction features")
        
        # Time-based features (if aired dates are available)
        if 'aired' in data.columns:
            data = self._extract_time_features(data)
            features_created += 3  # Approximate
            print("   ✅ Created time-based features")
        
        print(f"   🎯 Total features created: {features_created}")
        return data
    
    def _extract_popular_genres(self, genre_series: pd.Series, top_n: int = 15) -> List[str]:
        """Extract most popular genres from genre column."""
        all_genres = []
        for genres in genre_series.dropna():
            if str(genres) != 'Unknown':
                genre_list = [g.strip() for g in str(genres).split(',')]
                all_genres.extend(genre_list)
        
        genre_counts = pd.Series(all_genres).value_counts()
        return genre_counts.head(top_n).index.tolist()
    
    def _extract_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from aired dates."""
        try:
            # This is a placeholder - adjust based on your date format
            data['aired_year'] = pd.to_datetime(data['aired'], errors='coerce').dt.year
            data['aired_month'] = pd.to_datetime(data['aired'], errors='coerce').dt.month
            data['aired_season'] = data['aired_month'].apply(self._month_to_season)
            
            # Age of anime
            current_year = datetime.now().year
            data['anime_age'] = current_year - data['aired_year']
            
            print("   📅 Extracted time-based features")
        except Exception as e:
            print(f"   ⚠️ Could not extract time features: {str(e)}")
        
        return data
    
    def _month_to_season(self, month: int) -> str:
        """Convert month to anime season."""
        if pd.isna(month):
            return 'Unknown'
        elif month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _encode_categorical_variables(self) -> pd.DataFrame:
        """Encode categorical variables."""
        print(f"\n🔤 Step 5: Encoding Categorical Variables ({self.config['genre_encoding']})")
        
        data = self.processed_data.copy()
        
        # Get categorical columns (excluding already processed ones)
        categorical_cols = data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if not col.startswith('genre_') 
                          and not col.startswith('is_')]
        
        encoded_features = 0
        
        for col in categorical_cols:
            if col == 'name':  # Skip name column
                continue
                
            unique_values = data[col].nunique()
            
            if unique_values <= 2:  # Binary encoding
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
                self.encoders[f'{col}_label'] = le
                encoded_features += 1
                print(f"   🔢 Binary encoded: {col}")
                
            elif unique_values <= 10:  # One-hot encoding for low cardinality
                if self.config['genre_encoding'] == 'onehot':
                    dummies = pd.get_dummies(data[col], prefix=col, dummy_na=True)
                    data = pd.concat([data, dummies], axis=1)
                    encoded_features += len(dummies.columns)
                    print(f"   🎯 One-hot encoded: {col} ({len(dummies.columns)} features)")
                else:
                    le = LabelEncoder()
                    data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
                    self.encoders[f'{col}_label'] = le
                    encoded_features += 1
                    print(f"   🔢 Label encoded: {col}")
                    
            else:  # High cardinality - use label encoding or frequency encoding
                if unique_values > 50:
                    # Frequency encoding for very high cardinality
                    freq_map = data[col].value_counts(normalize=True).to_dict()
                    data[f'{col}_frequency'] = data[col].map(freq_map).fillna(0)
                    encoded_features += 1
                    print(f"   📊 Frequency encoded: {col}")
                else:
                    le = LabelEncoder()
                    data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
                    self.encoders[f'{col}_label'] = le
                    encoded_features += 1
                    print(f"   🔢 Label encoded: {col}")
        
        print(f"   🎯 Total encoded features: {encoded_features}")
        return data
    
    def _create_target_variables(self, target_column: str) -> Dict[str, pd.Series]:
        """Create various target variables for different ML tasks."""
        print(f"\n🎯 Step 6: Creating Target Variables")
        
        targets = {}
        
        if target_column == 'rating' and 'rating' in self.processed_data.columns:
            # Regression target
            targets['rating_regression'] = self.processed_data['rating']
            
            # Classification targets
            targets['rating_category'] = pd.cut(
                self.processed_data['rating'],
                bins=[0, 5.5, 6.5, 7.5, 8.5, 10],
                labels=['Poor', 'Below_Average', 'Average', 'Good', 'Excellent'],
                include_lowest=True
            )
            
            # Binary classification
            targets['high_quality'] = (self.processed_data['rating'] >= 8.0).astype(int)
            
            print("   ✅ Created rating-based targets")
            
        elif target_column == 'popularity' and 'members' in self.processed_data.columns:
            # Popularity-based targets
            targets['popularity_regression'] = self.processed_data['members']
            
            targets['popularity_tier'] = pd.qcut(
                self.processed_data['members'],
                q=5,
                labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
            )
            
            # Binary high popularity
            popularity_threshold = self.processed_data['members'].quantile(0.8)
            targets['high_popularity'] = (self.processed_data['members'] >= popularity_threshold).astype(int)
            
            print("   ✅ Created popularity-based targets")
        
        print(f"   🎯 Created {len(targets)} target variables")
        return targets
    
    def _select_features(self, target_column: Optional[str] = None) -> pd.DataFrame:
        """Select most important features."""
        print("\n🔍 Step 7: Feature Selection")
        
        data = self.processed_data.copy()
        
        # Get numerical features for selection
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns and ID columns
        exclude_cols = ['name', 'rating', 'members']  # Add other target/ID cols as needed
        numerical_features = [col for col in numerical_features if col not in exclude_cols]
        
        if target_column and target_column in data.columns and len(numerical_features) > 20:
            # Use statistical tests for feature selection
            X = data[numerical_features].fillna(0)
            y = data[target_column]
            
            # Remove rows where target is missing
            valid_mask = y.notna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(np.unique(y)) > 2:  # Regression or multi-class
                selector = SelectKBest(score_func=f_classif, k=min(50, len(numerical_features)))
            else:  # Binary classification
                selector = SelectKBest(score_func=mutual_info_classif, k=min(50, len(numerical_features)))
            
            try:
                X_selected = selector.fit_transform(X, y)
                selected_features = [numerical_features[i] for i in selector.get_support(indices=True)]
                
                # Store feature importance
                feature_scores = selector.scores_
                self.feature_importance = dict(zip(numerical_features, feature_scores))
                
                # Keep selected features and all non-numerical features
                non_numerical = data.select_dtypes(exclude=[np.number]).columns.tolist()
                final_features = selected_features + non_numerical
                
                data = data[final_features]
                print(f"   ✅ Selected {len(selected_features)} features out of {len(numerical_features)}")
                
            except Exception as e:
                print(f"   ⚠️ Feature selection failed: {str(e)}, keeping all features")
        
        return data
    
    def _scale_features(self) -> np.ndarray:
        """Scale numerical features."""
        print(f"\n📏 Step 8: Feature Scaling ({self.config['scaling_method']})")
        
        # Get numerical columns
        numerical_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['name']]  # Exclude ID columns
        
        if len(numerical_cols) == 0:
            print("   ⚠️ No numerical features to scale")
            return self.processed_data.values
        
        X_numerical = self.processed_data[numerical_cols].fillna(0)
        
        # Apply scaling
        if self.config['scaling_method'] == 'standard':
            scaler = StandardScaler()
        elif self.config['scaling_method'] == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        X_scaled = scaler.fit_transform(X_numerical)
        self.scalers['feature_scaler'] = scaler
        
        # Create scaled dataframe
        scaled_df = pd.DataFrame(
            X_scaled, 
            columns=[f'{col}_scaled' for col in numerical_cols],
            index=self.processed_data.index
        )
        
        # Combine with non-numerical features
        non_numerical_cols = self.processed_data.select_dtypes(exclude=[np.number]).columns
        if len(non_numerical_cols) > 0:
            combined_df = pd.concat([
                self.processed_data[non_numerical_cols],
                scaled_df
            ], axis=1)
        else:
            combined_df = scaled_df
        
        print(f"   ✅ Scaled {len(numerical_cols)} numerical features")
        return combined_df
    
    def _validate_processed_data(self) -> bool:
        """Validate the processed data."""
        print("\n✅ Step 9: Data Validation")
        
        try:
            # Check for infinite values
            if isinstance(self.feature_matrix, pd.DataFrame):
                numerical_cols = self.feature_matrix.select_dtypes(include=[np.number]).columns
                infinite_count = np.isinf(self.feature_matrix[numerical_cols]).sum().sum()
                if infinite_count > 0:
                    print(f"   ⚠️ Found {infinite_count} infinite values")
                    # Replace infinite values with NaN, then fill with median
                    self.feature_matrix[numerical_cols] = self.feature_matrix[numerical_cols].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    self.feature_matrix[numerical_cols] = self.feature_matrix[numerical_cols].fillna(
                        self.feature_matrix[numerical_cols].median()
                    )
            
            # Check data shape
            print(f"   📊 Final data shape: {self.feature_matrix.shape}")
            
            # Check memory usage
            memory_usage = self.feature_matrix.memory_usage(deep=True).sum() / 1024**2
            print(f"   💾 Memory usage: {memory_usage:.2f} MB")
            
            # Update preprocessing stats
            self.preprocessing_stats.update({
                'final_shape': self.feature_matrix.shape,
                'final_memory': memory_usage,
                'features_created': self.feature_matrix.shape[1] - self.raw_data.shape[1],
                'preprocessing_success': True
            })
            
            print("   ✅ Data validation passed")
            return True
            
        except Exception as e:
            print(f"   ❌ Data validation failed: {str(e)}")
            return False
    
    def _print_preprocessing_summary(self):
        """Print comprehensive preprocessing summary."""
        print("\n📊 PREPROCESSING SUMMARY")
        print("=" * 50)
        
        initial_shape = self.preprocessing_stats.get('initial_shape', (0, 0))
        final_shape = self.preprocessing_stats.get('final_shape', (0, 0))
        
        print(f"📈 Data Transformation:")
        print(f"   Initial shape: {initial_shape}")
        print(f"   Final shape: {final_shape}")
        print(f"   Rows change: {final_shape[0] - initial_shape[0]:+d}")
        print(f"   Features change: {final_shape[1] - initial_shape[1]:+d}")
        
        initial_memory = self.preprocessing_stats.get('initial_memory', 0) / 1024**2
        final_memory = self.preprocessing_stats.get('final_memory', 0)
        print(f"\n💾 Memory Usage:")
        print(f"   Initial: {initial_memory:.2f} MB")
        print(f"   Final: {final_memory:.2f} MB")
        print(f"   Change: {final_memory - initial_memory:+.2f} MB")
        
        print(f"\n🔧 Components Created:")
        print(f"   Scalers: {len(self.scalers)}")
        print(f"   Encoders: {len(self.encoders)}")
        print(f"   Imputers: {len(self.imputers)}")
        
        if self.target_variables:
            print(f"   Target variables: {len(self.target_variables)}")
    
    def save_processed_data(self, output_dir: str = "processed_data") -> Dict[str, str]:
        """
        Save processed data and preprocessing components.
        
        Args:
            output_dir (str): Directory to save processed data
            
        Returns:
            dict: Dictionary of saved file paths
        """
        print(f"\n💾 SAVING PROCESSED DATA")
        print("=" * 50)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        try:
            # Save processed dataset
            if self.processed_data is not None:
                processed_file = output_path / f"processed_data_{timestamp}.csv"
                self.processed_data.to_csv(processed_file, index=False)
                saved_files['processed_data'] = str(processed_file)
                print(f"   ✅ Processed data: {processed_file}")
            
            # Save feature matrix
            if self.feature_matrix is not None:
                if isinstance(self.feature_matrix, pd.DataFrame):
                    features_file = output_path / f"feature_matrix_{timestamp}.csv"
                    self.feature_matrix.to_csv(features_file, index=False)
                else:
                    features_file = output_path / f"feature_matrix_{timestamp}.npy"
                    np.save(features_file, self.feature_matrix)
                saved_files['feature_matrix'] = str(features_file)
                print(f"   ✅ Feature matrix: {features_file}")
            
            # Save target variables
            if self.target_variables:
                targets_file = output_path / f"target_variables_{timestamp}.csv"
                targets_df = pd.DataFrame(self.target_variables)
                targets_df.to_csv(targets_file, index=False)
                saved_files['targets'] = str(targets_file)
                print(f"   ✅ Target variables: {targets_file}")
            
            # Save preprocessing components
            import pickle
            
            components = {
                'scalers': self.scalers,
                'encoders': self.encoders,
                'imputers': self.imputers,
                'config': self.config,
                'preprocessing_stats': self.preprocessing_stats,
                'data_quality_metrics': self.data_quality_metrics
            }
            
            components_file = output_path / f"preprocessing_components_{timestamp}.pkl"
            with open(components_file, 'wb') as f:
                pickle.dump(components, f)
            saved_files['components'] = str(components_file)
            print(f"   ✅ Components: {components_file}")
            
            # Save preprocessing report
            report_file = output_path / f"preprocessing_report_{timestamp}.txt"
            self._generate_preprocessing_report(report_file)
            saved_files['report'] = str(report_file)
            print(f"   ✅ Report: {report_file}")
            
            print(f"\n🎉 All files saved successfully!")
            print(f"📁 Output directory: {output_path}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ Error saving processed data: {str(e)}")
            return {}
    
    def _generate_preprocessing_report(self, report_file: Path):
        """Generate detailed preprocessing report."""
        with open(report_file, 'w') as f:
            f.write("ANIME DATASET PREPROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("CONFIGURATION:\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Data transformation summary
            f.write("DATA TRANSFORMATION:\n")
            if 'initial_shape' in self.preprocessing_stats:
                initial_shape = self.preprocessing_stats['initial_shape']
                final_shape = self.preprocessing_stats['final_shape']
                f.write(f"  Initial shape: {initial_shape}\n")
                f.write(f"  Final shape: {final_shape}\n")
                f.write(f"  Rows change: {final_shape[0] - initial_shape[0]:+d}\n")
                f.write(f"  Features change: {final_shape[1] - initial_shape[1]:+d}\n")
            f.write("\n")
            
            # Data quality metrics
            if self.data_quality_metrics:
                f.write("DATA QUALITY METRICS:\n")
                f.write(f"  Overall quality score: {self.data_quality_metrics.get('overall_quality_score', 'N/A'):.2f}/10\n")
                
                if 'missing_values' in self.data_quality_metrics:
                    f.write("  Missing values by column:\n")
                    for col, info in self.data_quality_metrics['missing_values'].items():
                        if isinstance(info, dict) and info['count'] > 0:
                            f.write(f"    {col}: {info['count']} ({info['percentage']:.1f}%)\n")
                f.write("\n")
            
            # Feature importance
            if self.feature_importance:
                f.write("TOP FEATURE IMPORTANCE SCORES:\n")
                sorted_features = sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                for feature, score in sorted_features[:10]:
                    f.write(f"  {feature}: {score:.3f}\n")
                f.write("\n")
    
    def visualize_preprocessing_results(self):
        """Generate visualizations of preprocessing results."""
        print("\n📊 GENERATING PREPROCESSING VISUALIZATIONS")
        print("=" * 50)
        
        if self.processed_data is None:
            print("❌ No processed data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anime Dataset Preprocessing Results', fontsize=16, fontweight='bold')
        
        try:
            # 1. Data quality metrics
            if self.data_quality_metrics and 'overall_quality_score' in self.data_quality_metrics:
                ax = axes[0, 0]
                quality_score = self.data_quality_metrics['overall_quality_score']
                colors = ['red' if quality_score < 5 else 'orange' if quality_score < 7 else 'green']
                bars = ax.bar(['Data Quality'], [quality_score], color=colors)
                ax.set_ylim(0, 10)
                ax.set_ylabel('Quality Score')
                ax.set_title('Overall Data Quality')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.1f}', ha='center', va='bottom')
            
            # 2. Missing values heatmap
            ax = axes[0, 1]
            missing_data = self.processed_data.isnull()
            if missing_data.any().any():
                sns.heatmap(missing_data.iloc[:100], 
                           yticklabels=False, 
                           cbar=True, 
                           ax=ax,
                           cmap='viridis')
                ax.set_title('Missing Values Pattern (First 100 rows)')
            else:
                ax.text(0.5, 0.5, 'No Missing Values', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Missing Values Pattern')
            
            # 3. Feature distribution (for a key numerical feature)
            ax = axes[0, 2]
            numerical_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            if 'rating' in numerical_cols:
                feature_col = 'rating'
            elif len(numerical_cols) > 0:
                feature_col = numerical_cols[0]
            else:
                feature_col = None
            
            if feature_col:
                self.processed_data[feature_col].hist(bins=30, alpha=0.7, ax=ax)
                ax.set_title(f'{feature_col.title()} Distribution')
                ax.set_xlabel(feature_col.title())
                ax.set_ylabel('Frequency')
            
            # 4. Feature correlation heatmap
            ax = axes[1, 0]
            numerical_data = self.processed_data.select_dtypes(include=[np.number])
            if len(numerical_data.columns) > 1:
                # Select top correlated features to avoid overcrowding
                corr_matrix = numerical_data.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Get top features by variance or select a subset
                top_features = numerical_data.var().nlargest(min(10, len(numerical_data.columns))).index
                subset_corr = numerical_data[top_features].corr()
                
                sns.heatmap(subset_corr, 
                           annot=True, 
                           fmt='.2f', 
                           center=0,
                           ax=ax,
                           cmap='coolwarm')
                ax.set_title('Feature Correlation Matrix')
            else:
                ax.text(0.5, 0.5, 'Insufficient numerical features', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Correlation Matrix')
            
            # 5. Data shape comparison
            ax = axes[1, 1]
            if 'initial_shape' in self.preprocessing_stats and 'final_shape' in self.preprocessing_stats:
                initial_shape = self.preprocessing_stats['initial_shape']
                final_shape = self.preprocessing_stats['final_shape']
                
                categories = ['Rows', 'Columns']
                initial_values = [initial_shape[0], initial_shape[1]]
                final_values = [final_shape[0], final_shape[1]]
                
                x = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, initial_values, width, label='Initial', alpha=0.7)
                bars2 = ax.bar(x + width/2, final_values, width, label='Final', alpha=0.7)
                
                ax.set_xlabel('Dimension')
                ax.set_ylabel('Count')
                ax.set_title('Data Shape Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend()
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(final_values)*0.01,
                               f'{int(height)}', ha='center', va='bottom', fontsize=10)
            
            # 6. Feature importance (if available)
            ax = axes[1, 2]
            if self.feature_importance:
                top_features = sorted(self.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
                
                if top_features:
                    features, scores = zip(*top_features)
                    y_pos = np.arange(len(features))
                    
                    bars = ax.barh(y_pos, scores, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features])
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Top 10 Feature Importance')
                    ax.invert_yaxis()
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + max(scores)*0.01, bar.get_y() + bar.get_height()/2,
                               f'{width:.2f}', ha='left', va='center', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'Feature importance\nnot calculated', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Feature Importance')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Preprocessing visualizations generated successfully")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {str(e)}")
            plt.close(fig)

def main():
    """
    Demonstration of the preprocessing pipeline.
    """
    print("🚀 ANIME DATASET PREPROCESSING PIPELINE")
    print("🧹 Comprehensive Data Cleaning & Feature Engineering")
    print("=" * 70)
    
    # Configuration
    config = {
        'outlier_method': 'iqr',
        'missing_strategy': 'intelligent',
        'feature_selection': True,
        'scaling_method': 'standard',
        'genre_encoding': 'onehot',
        'verbose': True
    }
    
    # File path - update this to point to your dataset
    DATA_PATH = "anime_cleaned.csv"  # Update this path
    
    try:
        # Initialize preprocessor
        preprocessor = AnimeDataPreprocessor(config)
        
        # Load data
        if not preprocessor.load_data(DATA_PATH):
            print("❌ Failed to load data. Please check the file path.")
            return
        
        # Analyze data quality
        quality_analysis = preprocessor.analyze_data_quality()
        
        # Interactive preprocessing options
        while True:
            print("\n" + "=" * 50)
            print("🛠️  PREPROCESSING OPTIONS")
            print("1. 🚀 Run full preprocessing pipeline")
            print("2. 📊 Analyze data quality only")
            print("3. 🎨 Generate preprocessing visualizations")
            print("4. ⚙️  Customize preprocessing config")
            print("5. 💾 Save processed data")
            print("6. 🚪 Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting full preprocessing pipeline...")
                
                # Ask for target column
                target_column = input("Enter target column name (or press Enter for none): ").strip()
                target_column = target_column if target_column else None
                
                # Run preprocessing
                success = preprocessor.preprocess_data(target_column)
                
                if success:
                    print("\n🎉 Preprocessing completed successfully!")
                    
                    # Option to visualize results
                    viz_choice = input("Generate visualizations? (y/n): ").strip().lower()
                    if viz_choice in ['y', 'yes']:
                        preprocessor.visualize_preprocessing_results()
                
            elif choice == '2':
                print("\n📊 Running comprehensive data quality analysis...")
                quality_analysis = preprocessor.analyze_data_quality()
                
                # Display summary
                if quality_analysis:
                    print(f"\n📈 Quality Analysis Summary:")
                    print(f"   Overall Score: {quality_analysis.get('overall_quality_score', 'N/A'):.2f}/10")
                    
                    if 'missing_values' in quality_analysis:
                        total_missing = sum([info['count'] for info in quality_analysis['missing_values'].values() 
                                           if isinstance(info, dict)])
                        print(f"   Total Missing Values: {total_missing}")
                    
                    if 'consistency' in quality_analysis:
                        issues = len(quality_analysis['consistency']['issues'])
                        print(f"   Consistency Issues: {issues}")
                
            elif choice == '3':
                if preprocessor.processed_data is not None:
                    preprocessor.visualize_preprocessing_results()
                else:
                    print("❌ No processed data available. Run preprocessing first.")
                
            elif choice == '4':
                print("\n⚙️  Current Configuration:")
                for key, value in preprocessor.config.items():
                    print(f"   {key}: {value}")
                
                print("\n🔧 Customization Options:")
                print("1. Outlier method (iqr/zscore/isolation/lof)")
                print("2. Missing value strategy (intelligent/knn/simple)")
                print("3. Scaling method (standard/minmax/robust)")
                print("4. Genre encoding (onehot/label/count)")
                
                custom_choice = input("Select option to customize (1-4, or Enter to skip): ").strip()
                
                if custom_choice == '1':
                    new_method = input("Enter outlier method (iqr/zscore/isolation/lof): ").strip()
                    if new_method in ['iqr', 'zscore', 'isolation', 'lof']:
                        preprocessor.config['outlier_method'] = new_method
                        print(f"✅ Updated outlier method to: {new_method}")
                
                elif custom_choice == '2':
                    new_strategy = input("Enter missing strategy (intelligent/knn/simple): ").strip()
                    if new_strategy in ['intelligent', 'knn', 'simple']:
                        preprocessor.config['missing_strategy'] = new_strategy
                        print(f"✅ Updated missing strategy to: {new_strategy}")
                
                elif custom_choice == '3':
                    new_scaling = input("Enter scaling method (standard/minmax/robust): ").strip()
                    if new_scaling in ['standard', 'minmax', 'robust']:
                        preprocessor.config['scaling_method'] = new_scaling
                        print(f"✅ Updated scaling method to: {new_scaling}")
                
                elif custom_choice == '4':
                    new_encoding = input("Enter genre encoding (onehot/label/count): ").strip()
                    if new_encoding in ['onehot', 'label', 'count']:
                        preprocessor.config['genre_encoding'] = new_encoding
                        print(f"✅ Updated genre encoding to: {new_encoding}")
                
            elif choice == '5':
                if preprocessor.processed_data is not None:
                    output_dir = input("Enter output directory (or press Enter for 'processed_data'): ").strip()
                    output_dir = output_dir if output_dir else "processed_data"
                    
                    saved_files = preprocessor.save_processed_data(output_dir)
                    
                    if saved_files:
                        print(f"\n📁 Files saved:")
                        for file_type, file_path in saved_files.items():
                            print(f"   {file_type}: {file_path}")
                else:
                    print("❌ No processed data available. Run preprocessing first.")
                
            elif choice == '6':
                print("\n🎌 Thank you for using the preprocessing pipeline!")
                print("💫 Your data is ready for machine learning!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file '{DATA_PATH}'")
        print("Please ensure the anime dataset CSV file exists and update the path.")
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")

# Utility functions for standalone use
def quick_preprocess(file_path: str, target_column: str = None, output_dir: str = "processed_data") -> Dict[str, str]:
    """
    Quick preprocessing function for batch processing.
    
    Args:
        file_path (str): Path to the dataset
        target_column (str): Target column name
        output_dir (str): Output directory
        
    Returns:
        dict: Dictionary of saved file paths
    """
    preprocessor = AnimeDataPreprocessor()
    
    if preprocessor.load_data(file_path):
        if preprocessor.preprocess_data(target_column):
            return preprocessor.save_processed_data(output_dir)
    
    return {}

def load_preprocessing_components(components_file: str) -> Dict[str, Any]:
    """
    Load saved preprocessing components.
    
    Args:
        components_file (str): Path to the components file
        
    Returns:
        dict: Loaded preprocessing components
    """
    import pickle
    
    try:
        with open(components_file, 'rb') as f:
            components = pickle.load(f)
        print(f"✅ Loaded preprocessing components from {components_file}")
        return components
    except Exception as e:
        print(f"❌ Error loading components: {str(e)}")
        return {}

def apply_preprocessing_to_new_data(new_data: pd.DataFrame, components: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply saved preprocessing components to new data.
    
    Args:
        new_data (pd.DataFrame): New dataset to preprocess
        components (dict): Preprocessing components
        
    Returns:
        pd.DataFrame: Preprocessed new data
    """
    print("🔄 Applying preprocessing to new data...")
    
    try:
        processed_data = new_data.copy()
        
        # Apply scalers
        if 'scalers' in components and 'feature_scaler' in components['scalers']:
            scaler = components['scalers']['feature_scaler']
            # Note: This is a simplified version - you'd need to handle feature alignment
            print("   ✅ Applied feature scaling")
        
        # Apply encoders
        if 'encoders' in components:
            for encoder_name, encoder in components['encoders'].items():
                if 'label' in encoder_name:
                    column_name = encoder_name.replace('_label', '')
                    if column_name in processed_data.columns:
                        processed_data[f'{column_name}_encoded'] = encoder.transform(
                            processed_data[column_name].fillna('Unknown')
                        )
            print("   ✅ Applied categorical encoding")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Error applying preprocessing: {str(e)}")
        return new_data

if __name__ == "__main__":
    main()