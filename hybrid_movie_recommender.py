import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class HybridMovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_profiles = {}
        self.content_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.user_item_matrix = None
        self.knn_model = None
        self.genre_list = []
        
    def load_data(self, movies_path=None, ratings_path=None):
        """Load movie and ratings data"""
        if movies_path and ratings_path:
            self.movies_df = pd.read_csv(movies_path)
            self.ratings_df = pd.read_csv(ratings_path)
        else:
            # Create sample data if no data provided
            self._create_sample_data()
        
        self._preprocess_data()
        print("Data loaded and preprocessed successfully!")
    
    def _create_sample_data(self):
        """Create sample movie and rating data for demonstration"""
        # Sample movies data
        movies_data = {
            'movieId': range(1, 101),
            'title': [f'Movie {i}' for i in range(1, 101)],
            'genres': [
                'Action|Adventure', 'Comedy|Romance', 'Drama|Thriller', 'Horror|Mystery',
                'Sci-Fi|Fantasy', 'Action|Crime', 'Comedy|Family', 'Drama|Romance',
                'Horror|Thriller', 'Adventure|Fantasy'
            ] * 10,
            'overview': [
                f'This is an exciting movie {i} with great plot and characters.'
                for i in range(1, 101)
            ],
            'release_year': np.random.choice(range(1990, 2024), 100),
            'director': [f'Director {i%20}' for i in range(1, 101)],
            'cast': [f'Actor {i%30}, Actress {(i+1)%30}' for i in range(1, 101)]
        }
        
        # Sample ratings data
        np.random.seed(42)
        num_ratings = 5000
        ratings_data = {
            'userId': np.random.choice(range(1, 201), num_ratings),
            'movieId': np.random.choice(range(1, 101), num_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], num_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3])
        }
        
        self.movies_df = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data)
        
        # Remove duplicate user-movie pairs, keep the last rating
        self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'], keep='last')
        
        print("Sample data created successfully!")
    
    def _preprocess_data(self):
        """Preprocess the movie and ratings data"""
        # Extract unique genres
        all_genres = set()
        for genres in self.movies_df['genres'].dropna():
            all_genres.update(genres.split('|'))
        self.genre_list = sorted(list(all_genres))
        
        # Create genre binary matrix
        genre_matrix = pd.DataFrame(0, index=self.movies_df.index, columns=self.genre_list)
        for idx, genres in enumerate(self.movies_df['genres'].dropna()):
            for genre in genres.split('|'):
                if genre in self.genre_list:
                    genre_matrix.loc[idx, genre] = 1
        
        # Add genre columns to movies dataframe
        self.movies_df = pd.concat([self.movies_df, genre_matrix], axis=1)
        
        # Fill missing values
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['director'] = self.movies_df['director'].fillna('Unknown')
        self.movies_df['cast'] = self.movies_df['cast'].fillna('Unknown')
        
        # Create content features for content-based filtering
        self.movies_df['content_features'] = (
            self.movies_df['genres'].fillna('') + ' ' +
            self.movies_df['overview'] + ' ' +
            self.movies_df['director'] + ' ' +
            self.movies_df['cast']
        )
    
    def user_onboarding(self, user_id, preferred_genres, favorite_movies=None):
        """Onboard new user by collecting preferences"""
        if not isinstance(preferred_genres, list):
            preferred_genres = [preferred_genres]
        
        user_profile = {
            'user_id': user_id,
            'preferred_genres': preferred_genres,
            'watchlist': [],
            'watch_history': [],
            'ratings': {},
            'profile_created': True
        }
        
        if favorite_movies:
            user_profile['favorite_movies'] = favorite_movies
        
        self.user_profiles[user_id] = user_profile
        print(f"User {user_id} onboarded successfully with preferences: {preferred_genres}")
        return user_profile
    
    def add_to_watchlist(self, user_id, movie_id):
        """Add movie to user's watchlist"""
        if user_id in self.user_profiles:
            if movie_id not in self.user_profiles[user_id]['watchlist']:
                self.user_profiles[user_id]['watchlist'].append(movie_id)
                print(f"Movie {movie_id} added to user {user_id}'s watchlist")
    
    def add_rating(self, user_id, movie_id, rating):
        """Add user rating for a movie"""
        if user_id in self.user_profiles:
            self.user_profiles[user_id]['ratings'][movie_id] = rating
            if movie_id not in self.user_profiles[user_id]['watch_history']:
                self.user_profiles[user_id]['watch_history'].append(movie_id)
        
        # Also add to ratings dataframe
        new_rating = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'rating': [rating]
        })
        self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
        self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'], keep='last')
    
    def build_content_similarity_matrix(self):
        """Build content-based similarity matrix using TF-IDF"""
        print("Building content similarity matrix...")
        
        # Use TF-IDF on content features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['content_features'])
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print("Content similarity matrix built successfully!")
    
    def build_collaborative_filtering_model(self):
        """Build collaborative filtering model using KNN"""
        print("Building collaborative filtering model...")
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Convert to sparse matrix for efficiency
        user_item_sparse = csr_matrix(self.user_item_matrix.values)
        
        # Build KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=10,
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(user_item_sparse)
        
        print("Collaborative filtering model built successfully!")
    
    def content_based_recommendations(self, user_id, n_recommendations=10):
        """Generate content-based recommendations"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Get movies user has already watched
        watched_movies = set(user_profile['watch_history'])
        
        # Calculate genre preference scores
        genre_scores = {}
        for genre in user_profile['preferred_genres']:
            if genre in self.genre_list:
                genre_scores[genre] = 1.0
        
        # Score movies based on genre preferences
        movie_scores = []
        for idx, movie in self.movies_df.iterrows():
            if movie['movieId'] in watched_movies:
                continue
            
            score = 0
            for genre in self.genre_list:
                if genre in genre_scores and movie[genre] == 1:
                    score += genre_scores[genre]
            
            # Add content similarity bonus if user has watch history
            if user_profile['watch_history']:
                content_bonus = 0
                for watched_movie_id in user_profile['watch_history']:
                    watched_idx = self.movies_df[self.movies_df['movieId'] == watched_movie_id].index
                    if len(watched_idx) > 0:
                        watched_idx = watched_idx[0]
                        content_bonus += self.content_similarity_matrix[idx][watched_idx]
                score += content_bonus / len(user_profile['watch_history'])
            
            movie_scores.append((movie['movieId'], score))
        
        # Sort by score and return top recommendations
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in movie_scores[:n_recommendations]]
    
    def collaborative_filtering_recommendations(self, user_id, n_recommendations=10):
        """Generate collaborative filtering recommendations"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's rating vector
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(user_ratings, n_neighbors=11)
        similar_users = indices.flatten()[1:]  # Exclude the user themselves
        
        # Get movies rated by similar users
        recommendations = {}
        user_watched = set(self.user_item_matrix.columns[self.user_item_matrix.iloc[user_idx] > 0])
        
        for similar_user_idx in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]
            for movie_id, rating in similar_user_ratings.items():
                if rating > 3 and movie_id not in user_watched:  # Only consider good ratings
                    if movie_id not in recommendations:
                        recommendations[movie_id] = 0
                    recommendations[movie_id] += rating
        
        # Sort by accumulated rating scores
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [movie_id for movie_id, _ in sorted_recommendations[:n_recommendations]]
    
    def hybrid_recommendations(self, user_id, n_recommendations=10, content_weight=0.4, collab_weight=0.6):
        """Generate hybrid recommendations combining both approaches"""
        
        # Adjust weights based on user data availability
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # For new users (cold start), prefer content-based
        if len(user_profile['watch_history']) < 5:
            content_weight = 0.8
            collab_weight = 0.2
        else:
            content_weight = 0.3
            collab_weight = 0.7
        
        # Get recommendations from both methods
        content_recs = self.content_based_recommendations(user_id, n_recommendations * 2)
        collab_recs = self.collaborative_filtering_recommendations(user_id, n_recommendations * 2)
        
        # Combine recommendations with weights
        combined_scores = {}
        
        # Score content-based recommendations
        for i, movie_id in enumerate(content_recs):
            score = content_weight * (len(content_recs) - i) / len(content_recs)
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + score
        
        # Score collaborative filtering recommendations
        for i, movie_id in enumerate(collab_recs):
            score = collab_weight * (len(collab_recs) - i) / len(collab_recs)
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + score
        
        # Sort by combined scores
        sorted_recommendations = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [movie_id for movie_id, _ in sorted_recommendations[:n_recommendations]]
    
    def evaluate_recommendations(self, test_users, method='hybrid'):
        """Evaluate recommendation system performance"""
        precisions = []
        recalls = []
        
        for user_id in test_users:
            if user_id not in self.user_profiles:
                continue
            
            # Get actual liked movies (rating >= 4)
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            actual_liked = set(user_ratings[user_ratings['rating'] >= 4]['movieId'].tolist())
            
            if len(actual_liked) == 0:
                continue
            
            # Get recommendations
            if method == 'content':
                recommendations = self.content_based_recommendations(user_id, 10)
            elif method == 'collaborative':
                recommendations = self.collaborative_filtering_recommendations(user_id, 10)
            else:  # hybrid
                recommendations = self.hybrid_recommendations(user_id, 10)
            
            if len(recommendations) == 0:
                continue
            
            # Calculate precision and recall
            recommended_set = set(recommendations)
            relevant_recommended = actual_liked.intersection(recommended_set)
            
            precision = len(relevant_recommended) / len(recommended_set) if len(recommended_set) > 0 else 0
            recall = len(relevant_recommended) / len(actual_liked) if len(actual_liked) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1
        }
    
    def display_recommendations(self, user_id, method='hybrid', n_recommendations=10):
        """Display recommendations for a user"""
        if method == 'content':
            recommendations = self.content_based_recommendations(user_id, n_recommendations)
            method_name = "Content-Based"
        elif method == 'collaborative':
            recommendations = self.collaborative_filtering_recommendations(user_id, n_recommendations)
            method_name = "Collaborative Filtering"
        else:
            recommendations = self.hybrid_recommendations(user_id, n_recommendations)
            method_name = "Hybrid"
        
        print(f"\n{method_name} Recommendations for User {user_id}:")
        print("-" * 50)
        
        if not recommendations:
            print("No recommendations available.")
            return
        
        for i, movie_id in enumerate(recommendations, 1):
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                movie = movie_info.iloc[0]
                print(f"{i}. {movie['title']} ({movie['release_year']})")
                print(f"   Genres: {movie['genres']}")
                print(f"   Director: {movie['director']}")
                print()
    
    def run_evaluation_comparison(self):
        """Run comprehensive evaluation comparing all methods"""
        print("Running evaluation comparison...")
        
        # Get test users (users with sufficient rating history)
        user_counts = self.ratings_df['userId'].value_counts()
        test_users = user_counts[user_counts >= 5].index.tolist()[:50]  # Limit for demo
        
        methods = ['content', 'collaborative', 'hybrid']
        results = {}
        
        for method in methods:
            print(f"Evaluating {method} method...")
            metrics = self.evaluate_recommendations(test_users, method)
            results[method] = metrics
        
        # Display results
        print("\nEvaluation Results:")
        print("=" * 60)
        for method, metrics in results.items():
            print(f"{method.capitalize()} Filtering:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print()
        
        return results
    
    def visualize_results(self, results):
        """Visualize evaluation results"""
        methods = list(results.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [results[method][metric] for method in methods]
            axes[i].bar(methods, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to demonstrate the recommendation system"""
    
    # Initialize the recommendation system
    recommender = HybridMovieRecommendationSystem()
    
    # Load data (using sample data for demonstration)
    print("Loading data...")
    recommender.load_data()
    
    # Build models
    print("\nBuilding recommendation models...")
    recommender.build_content_similarity_matrix()
    recommender.build_collaborative_filtering_model()
    
    # Demonstrate user onboarding (Cold Start Solution)
    print("\nDemonstrating User Onboarding...")
    
    # Onboard new users with preferences
    recommender.user_onboarding(999, ['Action', 'Adventure', 'Sci-Fi'])
    recommender.user_onboarding(998, ['Comedy', 'Romance'])
    recommender.user_onboarding(997, ['Horror', 'Thriller', 'Mystery'])
    
    # Add some ratings for established user behavior
    print("\nSimulating user interactions...")
    recommender.add_rating(999, 1, 5)  # User 999 rates Movie 1 highly
    recommender.add_rating(999, 5, 4)  # User 999 rates Movie 5 well
    recommender.add_rating(998, 2, 5)  # User 998 rates Movie 2 highly
    recommender.add_rating(998, 7, 4)  # User 998 rates Movie 7 well
    
    # Display recommendations for new users (Cold Start)
    print("\n" + "="*60)
    print("COLD START DEMONSTRATION")
    print("="*60)
    
    recommender.display_recommendations(999, 'content', 5)
    recommender.display_recommendations(999, 'collaborative', 5)
    recommender.display_recommendations(999, 'hybrid', 5)
    
    # Display recommendations for different users
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR DIFFERENT USERS")
    print("="*60)
    
    for user_id in [998, 997]:
        recommender.display_recommendations(user_id, 'hybrid', 5)
    
    # Run evaluation
    print("\n" + "="*60)
    print("SYSTEM EVALUATION")
    print("="*60)
    
    results = recommender.run_evaluation_comparison()
    
    # Visualize results
    try:
        recommender.visualize_results(results)
    except:
        print("Visualization skipped (matplotlib display not available)")
    
    print("\nDemo completed successfully!")
    return recommender

if __name__ == "__main__":
    recommender = main()