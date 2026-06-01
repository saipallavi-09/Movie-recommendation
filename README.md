# CineMatch: AI Content-Based Movie Recommender System

CineMatch is a modern, visually stunning web application designed to compute and display highly relevant movie recommendations based on genre similarity using a 62k+ movie index.

## 🚀 Key Features

* **AI Recommendation Engine**: Input text keywords to search a movie, select it, and dynamically receive recommended movies matching your selection's themes.
* **Set-Based Jaccard Similarity**: An optimized search-matching algorithm that filters disjoint sets first, then computes Jaccard similarity in less than 0.05 seconds.
* **Interactive Genre Analytics**: Interactive Altair bar charts showing the overall genre distributions across the dataset.
* **Premium Dark-Theme Design**: Deployed with hovering glassmorphism cards, colorful category tags, and glowing gradients.

## 💻 Setup & Execution

1. **Install Dependencies**:
   ```bash
   pip install streamlit pandas numpy altair
   ```

2. **Run the Server**:
   ```bash
   streamlit run app.py
   ```
   Open your browser to the local URL (defaulting to `http://localhost:8501`).
