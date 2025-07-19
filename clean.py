import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def clean_and_reduce_dataset(input_file, output_file, target_size=3000):
    """
    Clean the anime dataset and reduce it to the target size while maintaining quality and diversity
    """
    print(f"Loading dataset from {input_file}...")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        print(f"Original dataset info:")
        print(df.info())
        print(f"\nMissing values:\n{df.isnull().sum()}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    print("\n" + "="*60)
    print("STEP 1: DATA CLEANING")
    print("="*60)
    
    # 1. Remove rows with missing critical information
    original_size = len(df)
    
    # Remove rows with missing name (critical)
    df = df.dropna(subset=['name'])
    print(f"After removing rows with missing names: {len(df)} rows")
    
    # Remove rows with missing rating (important for recommendations)
    df = df.dropna(subset=['rating'])
    print(f"After removing rows with missing ratings: {len(df)} rows")
    
    # 2. Handle missing values in other columns
    df['genre'] = df['genre'].fillna('Unknown')
    df['type'] = df['type'].fillna('Unknown')
    df['episodes'] = df['episodes'].fillna(1)  # Default to 1 episode
    df['members'] = df['members'].fillna(0)
    
    print(f"Missing values after cleaning:\n{df.isnull().sum()}")
    
    # 3. Remove duplicates based on name (case-insensitive)
    before_dup = len(df)
    df['name_lower'] = df['name'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['name_lower'])
    df = df.drop('name_lower', axis=1)
    print(f"After removing duplicates: {len(df)} rows (removed {before_dup - len(df)} duplicates)")
    
    # 4. Data validation and cleaning
    # Remove invalid ratings (should be between 0-10)
    df = df[(df['rating'] >= 0) & (df['rating'] <= 10)]
    print(f"After removing invalid ratings: {len(df)} rows")
    
    # Convert episodes to numeric
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

    # (Optional) Remove rows with missing episodes
    df = df[df['episodes'].notnull()]

    # Now you can safely compare
    df = df[df['episodes'] >= 0]
    print(f"After removing negative episodes/members: {len(df)} rows")
    
    # 5. Clean genre column - fix HTML entities
    df['genre'] = df['genre'].str.replace('&#039;', "'", regex=False)
    df['genre'] = df['genre'].str.replace('&amp;', '&', regex=False)
    
    # 6. Clean name column - fix HTML entities
    df['name'] = df['name'].str.replace('&#039;', "'", regex=False)
    df['name'] = df['name'].str.replace('&amp;', '&', regex=False)
    
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING FOR SELECTION")
    print("="*60)
    
    # Create features for intelligent selection
    df['genre_count'] = df['genre'].apply(lambda x: len(str(x).split(', ')) if pd.notna(x) and str(x) != 'Unknown' else 0)
    df['popularity_score'] = df['members'] * df['rating']
    df['episode_category'] = pd.cut(df['episodes'], 
                                   bins=[0, 1, 12, 26, 50, float('inf')], 
                                   labels=['Movie/Special', 'Short', 'Standard', 'Long', 'Very Long'])
    
    print(f"Dataset after feature engineering: {df.shape}")
    
    print("\n" + "="*60)
    print("STEP 3: INTELLIGENT DATA REDUCTION")
    print("="*60)
    
    if len(df) <= target_size:
        print(f"Dataset already has {len(df)} rows, which is <= target size of {target_size}")
        final_df = df.copy()
    else:
        # Strategy: Maintain diversity while prioritizing quality
        
        # 1. Always keep the top-rated animes (top 10%)
        top_rated_count = int(target_size * 0.1)  # 10% for top-rated
        top_rated = df.nlargest(top_rated_count, 'rating')
        print(f"Selected {len(top_rated)} top-rated animes")
        
        # 2. Select popular animes (high members count) - 20%
        remaining_df = df[~df.index.isin(top_rated.index)]
        popular_count = int(target_size * 0.2)  # 20% for popular
        popular = remaining_df.nlargest(popular_count, 'members')
        print(f"Selected {len(popular)} popular animes")
        
        # 3. Ensure type diversity - 30%
        remaining_df = remaining_df[~remaining_df.index.isin(popular.index)]
        type_diversity_count = int(target_size * 0.3)  # 30% for type diversity
        
        # Sample proportionally from each type
        type_counts = remaining_df['type'].value_counts()
        type_samples = []
        
        for anime_type in type_counts.index:
            type_data = remaining_df[remaining_df['type'] == anime_type]
            # Calculate sample size proportional to type frequency
            sample_size = min(
                len(type_data),
                max(1, int(type_diversity_count * (type_counts[anime_type] / type_counts.sum())))
            )
            if sample_size > 0:
                # Sample the best rated from this type
                type_sample = type_data.nlargest(sample_size, 'rating')
                type_samples.append(type_sample)
        
        type_diversity = pd.concat(type_samples) if type_samples else pd.DataFrame()
        print(f"Selected {len(type_diversity)} animes for type diversity")
        
        # 4. Genre diversity - 25%
        remaining_df = remaining_df[~remaining_df.index.isin(type_diversity.index)]
        genre_diversity_count = int(target_size * 0.25)  # 25% for genre diversity
        
        # Get all unique genres
        all_genres = set()
        for genres in remaining_df['genre'].dropna():
            if str(genres) != 'Unknown':
                all_genres.update([g.strip() for g in str(genres).split(',')])
        
        # Sample animes to cover different genres
        genre_samples = []
        for genre in list(all_genres)[:20]:  # Top 20 genres
            genre_data = remaining_df[remaining_df['genre'].str.contains(genre, na=False)]
            if len(genre_data) > 0:
                # Take top 2 from each genre
                genre_sample = genre_data.nlargest(min(2, len(genre_data)), 'rating')
                genre_samples.append(genre_sample)
        
        genre_diversity = pd.concat(genre_samples) if genre_samples else pd.DataFrame()
        genre_diversity = genre_diversity.drop_duplicates().head(genre_diversity_count)
        print(f"Selected {len(genre_diversity)} animes for genre diversity")
        
        # 5. Random sampling for the remaining slots - 15%
        remaining_df = remaining_df[~remaining_df.index.isin(genre_diversity.index)]
        remaining_count = target_size - len(top_rated) - len(popular) - len(type_diversity) - len(genre_diversity)
        
        if remaining_count > 0 and len(remaining_df) > 0:
            # Weighted random sampling (higher weight for better ratings)
            weights = remaining_df['rating'] / remaining_df['rating'].sum()
            random_sample = remaining_df.sample(
                n=min(remaining_count, len(remaining_df)), 
                weights=weights,
                random_state=42
            )
            print(f"Selected {len(random_sample)} animes through weighted random sampling")
        else:
            random_sample = pd.DataFrame()
        
        # Combine all selections
        final_df = pd.concat([top_rated, popular, type_diversity, genre_diversity, random_sample])
        final_df = final_df.drop_duplicates()  # Remove any duplicates
        
        # If still over target, randomly remove excess
        if len(final_df) > target_size:
            final_df = final_df.sample(n=target_size, random_state=42)
    
    # Clean up temporary columns
    columns_to_remove = ['genre_count', 'popularity_score', 'episode_category']
    final_df = final_df.drop(columns=[col for col in columns_to_remove if col in final_df.columns])
    
    # Reset index
    final_df = final_df.reset_index(drop=True)
    
    print("\n" + "="*60)
    print("STEP 4: FINAL DATASET STATISTICS")
    print("="*60)
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Reduction: {original_size} → {len(final_df)} rows ({(original_size-len(final_df))/original_size*100:.1f}% reduction)")
    
    print(f"\nFinal dataset statistics:")
    print(f"Rating range: {final_df['rating'].min():.2f} - {final_df['rating'].max():.2f}")
    print(f"Average rating: {final_df['rating'].mean():.2f}")
    print(f"Average episodes: {final_df['episodes'].mean():.1f}")
    print(f"Average members: {final_df['members'].mean():.0f}")
    
    print(f"\nType distribution:")
    print(final_df['type'].value_counts())
    
    print(f"\nRating distribution:")
    print(pd.cut(final_df['rating'], bins=5).value_counts().sort_index())
    
    # Save the cleaned dataset
    print(f"\nSaving cleaned dataset to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print(f"Dataset successfully saved!")
    
    return final_df

def main():
    """Main function to run the data cleaning process"""
    input_file = r"D:\AI assignment\anime.csv"
    output_file = r"D:\AI assignment\anime_cleaned.csv"
    target_size = 3000
    
    print("Anime Dataset Cleaning and Reduction Tool")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target size: {target_size} rows")
    print("=" * 60)
    
    # Run the cleaning process
    cleaned_df = clean_and_reduce_dataset(input_file, output_file, target_size)
    
    if cleaned_df is not None:
        print(f"\n✅ Process completed successfully!")
        print(f"Cleaned dataset saved as: {output_file}")
        print(f"Final size: {len(cleaned_df)} rows")
        
        # Show a sample of the cleaned data
        print(f"\nSample of cleaned data:")
        print(cleaned_df.head(10).to_string(index=False))
    else:
        print("❌ Process failed!")

if __name__ == "__main__":
    main()