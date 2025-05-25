#!/usr/bin/env python3
"""
Simple Demo Runner for Hybrid Movie Recommendation System
Run this script to see the system in action with sample data
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hybrid_movie_recommender import HybridMovieRecommendationSystem
    print("✓ Successfully imported recommendation system")
except ImportError as e:
    print(f"✗ Error importing recommendation system: {e}")
    print("Make sure hybrid_movie_recommender.py is in the same directory")
    sys.exit(1)

def quick_demo():
    """Run a quick demonstration of the system"""
    print("\n" + "="*60)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM - QUICK DEMO")
    print("="*60)
    
    # Initialize system
    print("\n1. Initializing recommendation system...")
    recommender = HybridMovieRecommendationSystem()
    
    # Load sample data
    print("2. Loading sample data...")
    recommender.load_data()
    
    # Build models
    print("3. Building recommendation models...")
    recommender.build_content_similarity_matrix()
    recommender.build_collaborative_filtering_model()
    
    # Demonstrate cold start solution
    print("\n4. Demonstrating Cold Start Solution...")
    print("-" * 40)
    
    # Create new users with different preferences
    new_users = [
        (1001, ['Action', 'Adventure', 'Sci-Fi'], "Action Movie Fan"),
        (1002, ['Comedy', 'Romance'], "Rom-Com Lover"),
        (1003, ['Horror', 'Thriller'], "Horror Enthusiast"),
        (1004, ['Drama', 'Biography'], "Drama Connoisseur")
    ]
    
    for user_id, preferences, description in new_users:
        print(f"\nOnboarding {description} (User {user_id})...")
        recommender.user_onboarding(user_id, preferences)
        
        # Show initial recommendations (cold start)
        print(f"\nInitial Recommendations for {description}:")
        recommender.display_recommendations(user_id, 'hybrid', 3)
    
    # Simulate user interactions
    print("\n5. Simulating User Interactions...")
    print("-" * 40)
    
    # Add some ratings to show how system evolves
    interactions = [
        (1001, [(1, 5), (5, 4), (15, 3)]),  # Action fan rates some movies
        (1002, [(2, 5), (7, 4), (12, 2)]),  # Comedy fan rates some movies
    ]
    
    for user_id, ratings in interactions:
        print(f"\nUser {user_id} rates some movies...")
        for movie_id, rating in ratings:
            recommender.add_rating(user_id, movie_id, rating)
            print(f"  Rated Movie {movie_id}: {rating}/5")
        
        # Show updated recommendations
        print(f"\nUpdated recommendations after ratings:")
        recommender.display_recommendations(user_id, 'hybrid', 3)
    
    # Compare different methods
    print("\n6. Comparing Recommendation Methods...")
    print("-" * 40)
    
    test_user = 1001
    print(f"\nRecommendations for User {test_user} using different methods:")
    
    methods = [
        ('content', 'Content-Based Filtering'),
        ('collaborative', 'Collaborative Filtering'),
        ('hybrid', 'Hybrid Approach')
    ]
    
    for method_code, method_name in methods:
        print(f"\n{method_name}:")
        try:
            recommender.display_recommendations(test_user, method_code, 3)
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
    
    # Run evaluation
    print("\n7. System Performance Evaluation...")
    print("-" * 40)
    
    try:
        results = recommender.run_evaluation_comparison()
        
        print("\nPerformance Summary:")
        print("-" * 30)
        best_precision = max(results.values(), key=lambda x: x['precision'])
        best_recall = max(results.values(), key=lambda x: x['recall'])
        best_f1 = max(results.values(), key=lambda x: x['f1_score'])
        
        for method, metrics in results.items():
            performance = "★" if metrics == best_f1 else " "
            print(f"{performance} {method.capitalize():15} - "
                  f"Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, "
                  f"F1: {metrics['f1_score']:.3f}")
        
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    print("\n8. Demo Complete!")
    print("="*60)
    print("Key Findings:")
    print("• Hybrid approach typically outperforms individual methods")
    print("• Cold start problem is effectively addressed through user onboarding")
    print("• System adapts and improves with user interactions")
    print("• Content-based filtering helps new users, collaborative filtering helps established users")
    
    return recommender

def interactive_demo():
    """Run an interactive demonstration"""
    print("\n" + "="*60)
    print("INTERACTIVE DEMO")
    print("="*60)
    
    recommender = HybridMovieRecommendationSystem()
    recommender.load_data()
    recommender.build_content_similarity_matrix()
    recommender.build_collaborative_filtering_model()
    
    print("\nAvailable genres:", ', '.join(recommender.genre_list))
    
    while True:
        print("\n" + "-"*40)
        print("What would you like to do?")
        print("1. Create new user profile")
        print("2. Get recommendations for existing user")
        print("3. Rate a movie")
        print("4. Compare recommendation methods")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            try:
                user_id = int(input("Enter user ID: "))
                print(f"Available genres: {', '.join(recommender.genre_list[:10])}...")
                genres_input = input("Enter preferred genres (comma-separated): ")
                preferred_genres = [g.strip() for g in genres_input.split(',')]
                
                recommender.user_onboarding(user_id, preferred_genres)
                recommender.display_recommendations(user_id, 'hybrid', 5)
                
            except ValueError:
                print("Please enter a valid user ID (number)")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            try:
                user_id = int(input("Enter user ID: "))
                if user_id in recommender.user_profiles:
                    recommender.display_recommendations(user_id, 'hybrid', 5)
                else:
                    print(f"User {user_id} not found. Please create profile first.")
            except ValueError:
                print("Please enter a valid user ID (number)")
        
        elif choice == '3':
            try:
                user_id = int(input("Enter user ID: "))
                movie_id = int(input("Enter movie ID: "))
                rating = int(input("Enter rating (1-5): "))
                
                if 1 <= rating <= 5:
                    recommender.add_rating(user_id, movie_id, rating)
                    print("Rating added successfully!")
                else:
                    print("Rating must be between 1 and 5")
            except ValueError:
                print("Please enter valid numbers")
        
        elif choice == '4':
            try:
                user_id = int(input("Enter user ID: "))
                if user_id in recommender.user_profiles:
                    for method in ['content', 'collaborative', 'hybrid']:
                        print(f"\n{method.upper()} RECOMMENDATIONS:")
                        recommender.display_recommendations(user_id, method, 3)
                else:
                    print(f"User {user_id} not found.")
            except ValueError:
                print("Please enter a valid user ID")
        
        elif choice == '5':
            print("Thank you for using the recommendation system!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")

def main():
    """Main function with menu options"""
    print("HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("Based on the research paper:")
    print("'Enhancing Movie Recommendations with a Hybrid Approach")
    print("Combining Content-Based and Collaborative Filtering'")
    print("\nChoose demo type:")
    print("1. Quick Demo (automated)")
    print("2. Interactive Demo")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            recommender = quick_demo()
            break
        elif choice == '2':
            interactive_demo()
            break
        elif choice == '3':
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your setup and try again.")