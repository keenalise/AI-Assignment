# 🎌 Anime Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An AI-powered anime recommendation system that helps users discover new anime based on their preferences using advanced machine learning techniques.

## 🚀 Project Overview

The Anime Recommendation System leverages machine learning algorithms to analyze anime characteristics and provide personalized recommendations. Built with Python and featuring both command-line and web interfaces, this system processes over 12,000 anime entries to deliver intelligent content discovery.

### 🎯 Key Features

- **🤖 AI-Powered Recommendations**: Uses TF-IDF vectorization and cosine similarity
- **📊 Hybrid Filtering**: Combines content-based filtering with rating analysis
- **🔍 Intelligent Clustering**: Groups similar anime using K-Means clustering
- **🌐 Web Interface**: Beautiful Streamlit-based GUI
- **💻 Command Line**: Traditional CLI for power users
- **📈 Data Analytics**: Comprehensive dataset analysis and visualization

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Framework**: Streamlit
- **Data Visualization**: matplotlib, seaborn, plotly
- **Text Processing**: TF-IDF Vectorization
- **Clustering**: K-Means Algorithm

## 📋 Prerequisites

Before running the system, ensure you have Python 3.8+ installed on your machine.

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/anime-recommendation-system.git
cd anime-recommendation-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Place your anime dataset at the following path:
```
D:\AI assignment\anime_cleaned.csv
```

### 4. Run the Application

#### Option A: Web Interface (Recommended)
```bash
streamlit run anime_app.py
```
Then open your browser to `http://localhost:8501`

#### Option B: Command Line Interface
```bash
python anime_recommendation_system.py
```

## 📁 Project Structure

```
📦 anime-recommendation-system/
├── 📄 anime_app.py                    # Streamlit web application
├── 📄 anime_recommendation_system.py  # Command-line interface
├── 📄 animebased.py                   # Core recommendation engine
├── 📄 clean.py                        # Data preprocessing utilities
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
├── 📄 anime_simple_report.txt         # System analysis report
└── 📊 D:\AI assignment\
    └── 📄 anime_cleaned.csv           # Dataset (place here)
```

## 🎮 How to Use

### Web Interface

1. **Load Dataset**: Upload or specify the dataset path in the sidebar
2. **Build Models**: Click "Build All AI Models" to train the recommendation engine
3. **Get Recommendations**: Enter an anime name and receive personalized suggestions
4. **Explore Data**: Browse statistics, search anime, and analyze trends

### Command Line Interface

1. **Start System**: Run the Python script and wait for model building
2. **Interactive Menu**: Choose from available options:
   - Get anime recommendations
   - View system statistics
   - Exit application
3. **Enter Preferences**: Input anime names to receive recommendations

## 🔧 System Architecture

### Data Processing Pipeline
```
Raw Dataset → Data Cleaning → Feature Engineering → Outlier Handling → Processed Data
```

### Recommendation Engine
```
TF-IDF Vectorization → Cosine Similarity → Hybrid Scoring → Final Recommendations
```

### Clustering Analysis
```
Feature Selection → Standardization → K-Means Clustering → Cluster Analysis
```

## 📊 Dataset Information

- **Total Entries**: 12,294 anime records
- **Features**: Name, Genre, Rating, Type, Episodes, Members
- **Rating Range**: 4.0 - 9.1
- **Average Rating**: 6.49
- **Data Quality**: Cleaned and preprocessed for optimal performance

### Dataset Schema
| Column | Type | Description |
|--------|------|-------------|
| name | string | Anime title |
| genre | string | Comma-separated genres |
| rating | float | User rating (0-10) |
| type | string | Anime type (TV, Movie, OVA, etc.) |
| episodes | integer | Number of episodes |
| members | integer | Number of community members |

## 🤖 Machine Learning Components

### 1. Content-Based Filtering
- **Algorithm**: TF-IDF Vectorization
- **Features**: Genre, Type, Episode categories
- **Similarity Metric**: Cosine Similarity
- **Matrix Size**: 12,000+ x 5,000 features

### 2. Hybrid Recommendation
- **Content Similarity**: 70% weight
- **Rating Quality**: 30% weight
- **Normalization**: Min-Max scaling
- **Output**: Top-N recommendations

### 3. Clustering Analysis
- **Algorithm**: K-Means
- **Optimal Clusters**: Determined by Silhouette Score
- **Features**: Rating, Episodes, Members, Popularity Score
- **Purpose**: Anime grouping and analysis

## 📈 Performance Metrics

- **Recommendation Speed**: < 1 second per query
- **Dataset Processing**: ~10 seconds for full preprocessing
- **Model Building**: ~5 seconds for TF-IDF matrix
- **Memory Usage**: ~50MB for processed dataset
- **Clustering Quality**: Silhouette Score > 0.3

## 🎨 Web Interface Features

### Dashboard Components
- **📊 Dataset Overview**: Key statistics and metrics
- **📈 Data Visualizations**: Interactive charts and graphs
- **🎯 Recommendation Engine**: Real-time anime suggestions
- **🔍 Data Explorer**: Search and browse functionality
- **📋 System Statistics**: Performance and analysis metrics

### Interactive Elements
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Dynamic content loading
- **Beautiful UI**: Modern glassmorphism design
- **Smooth Animations**: Enhanced user experience

## 🔧 Configuration

### Dataset Path
Update the dataset path in your chosen interface:

**For Streamlit App** (`anime_app.py`):
```python
default_path = st.text_input("Dataset path:", value="D:\\AI assignment\\anime_cleaned.csv")
```

**For CLI App** (`anime_recommendation_system.py`):
```python
system = AnimeRecommendationSystem("D:\\AI assignment\\anime_cleaned.csv")
```

### Model Parameters
Customize recommendation parameters:
```python
# TF-IDF Configuration
max_features = 5000
ngram_range = (1, 2)
min_df = 2
max_df = 0.9

# Recommendation Weights
content_weight = 0.7
rating_weight = 0.3
```

## 🚀 Advanced Usage

### Custom Dataset
To use your own anime dataset:
1. Ensure CSV format with required columns
2. Update the dataset path
3. Run the data cleaning script if needed:
```python
python clean.py
```

### Extending Functionality
Add new features by modifying:
- `_engineer_features()`: Add new calculated features
- `get_recommendations()`: Customize recommendation logic
- `build_clustering_model()`: Adjust clustering parameters

## 🐛 Troubleshooting

### Common Issues

**1. Dataset Not Found**
```
Error: CSV file not found. Please check the file path.
```
**Solution**: Verify the dataset path `D:\AI assignment\anime_cleaned.csv` exists

**2. Memory Issues**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce dataset size or increase system RAM

**3. Import Errors**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

**4. Streamlit Port Issues**
```
Port 8501 is already in use
```
**Solution**: Use different port: `streamlit run anime_app.py --server.port 8502`

## 📝 Requirements

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **MyAnimeList**: For providing anime data and ratings
- **scikit-learn**: For machine learning algorithms
- **Streamlit**: For the amazing web framework
- **Anime Community**: For ratings and reviews that make recommendations possible

## 📞 Contact

- **Author**: Keen Alise

---

<div align="center">

### 🎌 Built with ❤️ for the Anime Community

**Happy Anime Watching! 🍿✨**

</div>
