import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
from PIL import Image, ImageTk
import webbrowser

class AnimeRecommendationGUI:
    def __init__(self, root, recommendation_system):
        self.root = root
        self.system = recommendation_system
        self.setup_ui()
        
    def setup_ui(self):
        # Main window configuration
        self.root.title("Anime Recommendation System")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        self.style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'))
        self.style.configure('Error.TLabel', foreground='red')
        
        # Create main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.header_frame, text="Anime Recommendation System", 
                 style='Title.TLabel').pack(side=tk.LEFT)
        
        # Navigation tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_recommendation_tab()
        self.create_prediction_tab()
        self.create_search_tab()
        self.create_about_tab()
        
    def create_recommendation_tab(self):
        """Tab for getting anime recommendations"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Recommendations")
        
        # Main content frame
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input frame
        input_frame = ttk.Frame(content_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Get Anime Recommendations", style='Header.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Label(input_frame, text="Anime Name:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.rec_anime_entry = ttk.Entry(input_frame, width=40)
        self.rec_anime_entry.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        ttk.Label(input_frame, text="Number of Recommendations:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.rec_num_entry = ttk.Entry(input_frame, width=10)
        self.rec_num_entry.insert(0, "10")
        self.rec_num_entry.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Weights frame
        weights_frame = ttk.Frame(input_frame)
        weights_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        
        ttk.Label(weights_frame, text="Content Similarity Weight:").grid(row=0, column=0, sticky=tk.W)
        self.content_weight = ttk.Scale(weights_frame, from_=0, to=1, value=0.7, length=150)
        self.content_weight.grid(row=0, column=1, padx=5)
        self.content_weight_label = ttk.Label(weights_frame, text="0.7")
        self.content_weight_label.grid(row=0, column=2)
        
        ttk.Label(weights_frame, text="Rating Weight:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.rating_weight = ttk.Scale(weights_frame, from_=0, to=1, value=0.3, length=150)
        self.rating_weight.grid(row=1, column=1, padx=5, pady=(5, 0))
        self.rating_weight_label = ttk.Label(weights_frame, text="0.3")
        self.rating_weight_label.grid(row=1, column=2, pady=(5, 0))
        
        # Bind scale updates
        self.content_weight.bind("<Motion>", lambda e: self.update_scale_labels())
        self.rating_weight.bind("<Motion>", lambda e: self.update_scale_labels())
        
        # Button
        ttk.Button(input_frame, text="Get Recommendations", command=self.get_recommendations).grid(row=4, column=0, columnspan=2, pady=(15, 0))
        
        # Results frame
        results_frame = ttk.Frame(content_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for results
        self.rec_tree = ttk.Treeview(results_frame, columns=('name', 'genre', 'rating', 'type', 'episodes', 'similarity'), show='headings')
        
        # Configure columns
        self.rec_tree.heading('name', text='Name')
        self.rec_tree.heading('genre', text='Genre')
        self.rec_tree.heading('rating', text='Rating')
        self.rec_tree.heading('type', text='Type')
        self.rec_tree.heading('episodes', text='Episodes')
        self.rec_tree.heading('similarity', text='Similarity')
        
        # Set column widths
        self.rec_tree.column('name', width=200)
        self.rec_tree.column('genre', width=150)
        self.rec_tree.column('rating', width=60)
        self.rec_tree.column('type', width=60)
        self.rec_tree.column('episodes', width=60)
        self.rec_tree.column('similarity', width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.rec_tree.yview)
        self.rec_tree.configure(yscroll=scrollbar.set)
        
        # Pack widgets
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_prediction_tab(self):
        """Tab for predicting anime rating categories"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Rating Prediction")
        
        # Main content frame
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input frame
        input_frame = ttk.Frame(content_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Predict Anime Rating Category", style='Header.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Input fields
        fields = [
            ("Episodes:", "episodes_entry"),
            ("Expected Members:", "members_entry"),
            ("Number of Genres:", "genre_count_entry"),
            ("Type (TV/Movie/OVA/Special):", "type_entry"),
            ("Genres (comma-separated):", "genres_entry")
        ]
        
        for i, (label_text, attr_name) in enumerate(fields, start=1):
            ttk.Label(input_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=(10 if i == 1 else 5, 0))
            entry = ttk.Entry(input_frame, width=40)
            entry.grid(row=i, column=1, sticky=tk.W, pady=(10 if i == 1 else 5, 0))
            setattr(self, attr_name, entry)
        
        # Set default values for easier testing
        self.type_entry.insert(0, "TV")
        self.genres_entry.insert(0, "Action, Adventure")
        
        # Button
        ttk.Button(input_frame, text="Predict Category", command=self.predict_category).grid(row=len(fields)+1, column=0, columnspan=2, pady=(15, 0))
        
        # Results frame
        results_frame = ttk.Frame(content_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Prediction results
        self.prediction_frame = ttk.Frame(results_frame)
        self.prediction_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Probability distribution
        ttk.Label(results_frame, text="Category Probabilities:", style='Header.TLabel').pack(anchor=tk.W)
        
        self.prob_tree = ttk.Treeview(results_frame, columns=('category', 'probability'), show='headings')
        self.prob_tree.heading('category', text='Category')
        self.prob_tree.heading('probability', text='Probability')
        self.prob_tree.column('category', width=150)
        self.prob_tree.column('probability', width=100)
        self.prob_tree.pack(fill=tk.BOTH, expand=True)
        
    def create_search_tab(self):
        """Tab for searching anime in the database"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Search Anime")
        
        # Main content frame
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Search frame
        search_frame = ttk.Frame(content_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="Search Anime Database", style='Header.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Label(search_frame, text="Search Term:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.search_entry = ttk.Entry(search_frame, width=40)
        self.search_entry.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        ttk.Button(search_frame, text="Search", command=self.search_anime).grid(row=1, column=2, padx=(10, 0), pady=(10, 0))
        
        # Results frame
        results_frame = ttk.Frame(content_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for results
        self.search_tree = ttk.Treeview(results_frame, columns=('name', 'rating', 'type', 'episodes', 'genre'), show='headings')
        
        # Configure columns
        self.search_tree.heading('name', text='Name')
        self.search_tree.heading('rating', text='Rating')
        self.search_tree.heading('type', text='Type')
        self.search_tree.heading('episodes', text='Episodes')
        self.search_tree.heading('genre', text='Genre')
        
        # Set column widths
        self.search_tree.column('name', width=200)
        self.search_tree.column('rating', width=60)
        self.search_tree.column('type', width=60)
        self.search_tree.column('episodes', width=60)
        self.search_tree.column('genre', width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
        self.search_tree.configure(yscroll=scrollbar.set)
        
        # Pack widgets
        self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_about_tab(self):
        """Tab with information about the application"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="About")
        
        # Main content frame
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(content_frame, text="Anime Recommendation System", style='Title.TLabel').pack(pady=(0, 20))
        
        # Description
        desc = """
        This application provides intelligent anime recommendations using advanced 
        machine learning techniques including:
        
        • Content-based filtering with NLP (TF-IDF + N-grams)
        • Hybrid recommendation combining multiple algorithms
        • Ensemble classification models
        • Clustering analysis
        
        The system can:
        - Recommend similar anime based on content
        - Predict rating categories for new anime
        - Search through the anime database
        """
        
        ttk.Label(content_frame, text=desc, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 20))
        
        # Stats frame
        stats_frame = ttk.Frame(content_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Get some stats from the system
        try:
            total_animes = len(self.system.processed_data)
            avg_rating = self.system.processed_data['rating'].mean()
            unique_genres = len(set([g.strip() for genres in self.system.processed_data['genre'].dropna() 
                                  for g in str(genres).split(',') if g.strip()]))
            
            stats_text = f"""
            Dataset Statistics:
            • Total Animes: {total_animes:,}
            • Average Rating: {avg_rating:.2f}
            • Unique Genres: {unique_genres}
            """
            
            ttk.Label(stats_frame, text=stats_text, justify=tk.LEFT).pack(anchor=tk.W)
        except Exception as e:
            print(f"Error getting stats: {e}")
        
        # Footer
        ttk.Label(content_frame, text="Developed with Python and Tkinter", font=('Helvetica', 9, 'italic')).pack(side=tk.BOTTOM)
    
    def update_scale_labels(self):
        """Update the labels for the weight scales"""
        content_val = round(self.content_weight.get(), 1)
        rating_val = round(1 - content_val, 1)
        
        self.content_weight_label.config(text=str(content_val))
        self.rating_weight_label.config(text=str(rating_val))
        self.rating_weight.set(rating_val)
    
    def get_recommendations(self):
        """Get recommendations based on user input"""
        anime_name = self.rec_anime_entry.get().strip()
        if not anime_name:
            messagebox.showerror("Error", "Please enter an anime name")
            return
        
        try:
            num_recs = int(self.rec_num_entry.get())
            if num_recs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of recommendations")
            return
        
        content_weight = self.content_weight.get()
        rating_weight = self.rating_weight.get()
        
        # Clear previous results
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        # Get recommendations
        recommendations = self.system.get_hybrid_recommendations(
            anime_name, num_recs, content_weight, rating_weight
        )
        
        if isinstance(recommendations, str):
            messagebox.showerror("Error", recommendations)
            return
        
        # Add recommendations to treeview
        for _, row in recommendations.iterrows():
            self.rec_tree.insert('', tk.END, values=(
                row['name'],
                row['genre'],
                f"{row['rating']:.2f}",
                row['type'],
                row['episodes'],
                f"{row['content_similarity']:.3f}"
            ))
    
    def predict_category(self):
        """Predict anime rating category based on input features"""
        # Get input values
        try:
            episodes = int(self.episodes_entry.get())
            members = int(self.members_entry.get())
            genre_count = int(self.genre_count_entry.get())
            anime_type = self.type_entry.get().strip() or "TV"
            genres = self.genres_entry.get().strip()
            
            if episodes <= 0 or members <= 0 or genre_count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter valid positive numbers for episodes, members, and genre count")
            return
        
        # Clear previous results
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
        
        for item in self.prob_tree.get_children():
            self.prob_tree.delete(item)
        
        # Make prediction
        prediction = self.system.predict_rating_category_enhanced(
            episodes, members, genre_count, anime_type, genres
        )
        
        if isinstance(prediction, str):
            messagebox.showerror("Error", prediction)
            return
        
        # Display prediction results
        ttk.Label(self.prediction_frame, 
                 text=f"Predicted Category: {prediction['predicted_category']}",
                 font=('Helvetica', 12, 'bold')).pack(anchor=tk.W)
        
        ttk.Label(self.prediction_frame, 
                 text=f"Confidence: {prediction['confidence']:.1%}",
                 font=('Helvetica', 10)).pack(anchor=tk.W)
        
        ttk.Label(self.prediction_frame, 
                 text=f"Explanation: {prediction['explanation']}",
                 wraplength=700, justify=tk.LEFT).pack(anchor=tk.W, pady=(5, 0))
        
        ttk.Label(self.prediction_frame, 
                 text=f"Recommendation: {prediction['recommendation']}",
                 wraplength=700, justify=tk.LEFT).pack(anchor=tk.W, pady=(5, 0))
        
        # Add probabilities to treeview
        for category, prob in prediction['all_probabilities'].items():
            self.prob_tree.insert('', tk.END, values=(category, f"{prob:.1%}"))
    
    def search_anime(self):
        """Search anime in the database"""
        search_term = self.search_entry.get().strip()
        if not search_term:
            messagebox.showerror("Error", "Please enter a search term")
            return
        
        # Clear previous results
        for item in self.search_tree.get_children():
            self.search_tree.delete(item)
        
        # Search for anime
        matches = self.system.processed_data[
            self.system.processed_data['name'].str.contains(search_term, case=False, na=False)
        ].head(20)  # Limit to 20 results
        
        if len(matches) == 0:
            messagebox.showinfo("No Results", "No anime found matching your search term")
            return
        
        # Add results to treeview
        for _, row in matches.iterrows():
            self.search_tree.insert('', tk.END, values=(
                row['name'],
                f"{row['rating']:.2f}" if pd.notna(row['rating']) else "N/A",
                row['type'],
                row['episodes'],
                row['genre']
            ))

def main():
    """Main function to initialize the application"""
    # Initialize the recommendation system
    print("Initializing recommendation system...")
    try:
        anime_system = EnhancedAnimeRecommendationSystem(r"D:\AI assignment\anime_cleaned.csv")
        
        # Build required models
        anime_system.build_enhanced_content_recommender()
        anime_system.build_optimal_clustering_model()
        anime_system.build_ensemble_classifier()
        
        # Create and run GUI
        root = tk.Tk()
        app = AnimeRecommendationGUI(root, anime_system)
        root.mainloop()
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        messagebox.showerror("Fatal Error", f"Failed to initialize system: {str(e)}")

if __name__ == "__main__":
    main()