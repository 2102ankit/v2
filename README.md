
# 🎬 Hybrid Movie Recommender System

This project implements a hybrid movie recommendation system that combines content-based and collaborative filtering techniques to provide personalized movie suggestions.

## 📂 Project Structure

- **`hybrid_movie_recommender.py`**: Core module containing the implementation of the hybrid recommendation algorithm.
- **`run_demo.py`**: Script to demonstrate the functionality of the recommender system.
- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`setup_requirements.md`**: Instructions for setting up the development environment and installing dependencies.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Recommended: Use a virtual environment to manage dependencies.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/2102ankit/v2.git
   cd v2
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

Execute the demo script to see the recommender in action:

```bash
python run_demo.py
```

This will generate a list of recommended movies based on the hybrid algorithm.

## 🧠 How It Works

The hybrid recommender system integrates:

- **Content-Based Filtering**: Analyzes movie attributes (e.g., genres, directors, cast) to recommend similar movies.
- **Collaborative Filtering**: Utilizes user ratings and preferences to suggest movies liked by similar users.

By combining these methods, the system aims to improve recommendation accuracy and provide more personalized suggestions.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
